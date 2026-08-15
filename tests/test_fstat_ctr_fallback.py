"""Countable-only F-stat center cache + inline fallback + SNR-truncated
distance proposal (2026-08-15). Light fakes in the style of
test_rj_flip_fraction.py: no comps, no buffers — ``_fstat_dist_centers`` is
patched per instance with a deterministic stand-in.
"""

import os
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.special import erf, erfinv

from lisatools.globalfit.moves.gbspecialstretch import (
    GBSpecialStretchMove,
    cp as _cp,  # the module's RNG source (numpy on CPU-only machines)
)


def _phi(x):
    return 0.5 * (1.0 + erf(np.asarray(x) / np.sqrt(2.0)))


def _trapz(y, x):
    y = np.asarray(y)
    x = np.asarray(x)
    return float((0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])).sum())


def _fake_centers(model, params, walker_ref):
    """Deterministic per-row (A, phi0, iota, psi, F) from params."""
    f = np.asarray(params[:, 1], dtype=float)
    A = 1.0 + 0.01 * f
    phi0 = 0.1 * f
    iota = 0.2 + 0.001 * f
    psi = 0.3 + 0.002 * f
    F = 50.0 + f
    return A, phi0, iota, psi, F


def _move(basis="lnA"):
    m = GBSpecialStretchMove.__new__(GBSpecialStretchMove)
    m.name = "test"
    m.use_gpu = False
    m._backend_name = "lisatools_cpu"  # self.xp resolves through the backend
    # amplitude basis by default: _dist_center_and_width's distance branch
    # imports the stock transforms; the lnA branch is self-contained.
    m.transform_fn = SimpleNamespace(input_basis=[basis, "f0_ms", "fdot"])
    m._fstat_dist_centers = _fake_centers
    return m


def _sorter(num_sources, ndim=8):
    coords = np.zeros((num_sources, ndim))
    coords[:, 1] = 10.0 + np.arange(num_sources)  # unique deterministic f col
    coords[:, 0] = 1.0
    return SimpleNamespace(coords=coords, num_sources=num_sources)


def _subset(ids, alive):
    return SimpleNamespace(
        inds=np.asarray(alive, dtype=bool),
        inds_main_band_sorter=np.asarray(ids),
    )


class _SmearEnv(unittest.TestCase):
    def setUp(self):
        os.environ["GB_FSTAT_CTR_SMEAR"] = "1.5"
        self.addCleanup(os.environ.pop, "GB_FSTAT_CTR_SMEAR", None)


class CountablePrecomputeTest(_SmearEnv):
    def test_countable_only_contents(self):
        m = _move()
        sorter = _sorter(20)
        unit_ids = np.array([2, 5, 7, 9, 12])
        alive = np.array([True, False, False, True, False])
        # dead + at-cap (rows 7, 12) = the birth reserve -> excluded;
        # dead + below-cap (row 5) and every alive row stay.
        cap = np.zeros(20, dtype=bool)
        cap[[7, 12]] = True
        cap[9] = True  # alive at-cap row must STAY (deaths consume centers)
        m._rj_at_cap_mask = cap
        c = m._precompute_fstat_centers(None, sorter, _subset(unit_ids, alive))
        np.testing.assert_array_equal(c["ids"], [2, 5, 9])
        np.testing.assert_array_equal(c["unit_ids"], unit_ids)
        # values match the shared compute path row-for-row
        A, _, _, _, F = _fake_centers(None, sorter.coords[[2, 5, 9]], 0)
        np.testing.assert_allclose(c["ln_center"], np.log(A))
        snr = np.sqrt(np.clip(2.0 * F, 1.0, None))
        np.testing.assert_allclose(c["sigma"], 1.5 / snr)  # smear applied
        np.testing.assert_allclose(c["ln_snr"], np.log(snr))
        self.assertEqual(c["n_miss"], 0)

    def test_no_cap_mask_keeps_all_rows(self):
        m = _move()
        m._rj_at_cap_mask = None
        unit_ids = np.array([1, 4, 6])
        c = m._precompute_fstat_centers(
            None, _sorter(10), _subset(unit_ids, [True, False, False]))
        np.testing.assert_array_equal(c["ids"], unit_ids)

    def test_empty_unit_returns_none(self):
        m = _move()
        m._rj_at_cap_mask = None
        out = m._precompute_fstat_centers(
            None, _sorter(4), _subset(np.array([], dtype=int), []))
        self.assertIsNone(out)


class FallbackLookupTest(_SmearEnv):
    def _cache(self, m, sorter):
        unit_ids = np.array([2, 5, 7, 9, 12])
        alive = np.array([True, False, False, True, False])
        cap = np.zeros(sorter.num_sources, dtype=bool)
        cap[[7, 12]] = True
        m._rj_at_cap_mask = cap
        m._fstat_ctr = m._precompute_fstat_centers(
            None, sorter, _subset(unit_ids, alive))
        return m._fstat_ctr

    def test_hit_positions(self):
        m = _move()
        sorter = _sorter(20)
        c = self._cache(m, sorter)
        pos = m._fstat_ctr_lookup(np.array([9, 2]))
        np.testing.assert_array_equal(c["ids"][pos], [9, 2])

    def test_fallback_equals_direct_computation_and_appends(self):
        m = _move()
        sorter = _sorter(20)
        c = self._cache(m, sorter)
        calls = []
        orig = m._fstat_dist_centers

        def counting(model, params, walker_ref):
            calls.append(int(params.shape[0]))
            return orig(model, params, walker_ref)

        m._fstat_dist_centers = counting
        pos = m._fstat_ctr_lookup(
            np.array([7]), model=object(), band_sorter=sorter)
        self.assertEqual(calls, [1])  # exactly the missing row was computed
        np.testing.assert_array_equal(c["ids"], [2, 5, 7, 9])
        # fallback values == the shared compute path on that row (computed
        # on a FRESH move so the call counter above stays untouched)
        exp = _move()._fstat_ctr_compute(None, sorter.coords[[7]])
        for name, v in zip(m._FSTAT_CTR_FIELDS, exp):
            np.testing.assert_allclose(c[name][pos], v, err_msg=name)
        self.assertEqual(c["n_miss"], 1)
        # a row misses at most once: same lookup again -> no new compute
        m._fstat_ctr_lookup(np.array([7]), model=object(), band_sorter=sorter)
        self.assertEqual(calls, [1])

    def test_mixed_hit_and_miss_positions_consistent(self):
        m = _move()
        sorter = _sorter(20)
        c = self._cache(m, sorter)
        rows = np.array([9, 12, 2])
        pos = m._fstat_ctr_lookup(rows, model=object(), band_sorter=sorter)
        np.testing.assert_array_equal(c["ids"][pos], rows)
        # cache stayed sorted so later searchsorted lookups keep working
        np.testing.assert_array_equal(c["ids"], np.sort(c["ids"]))

    def test_foreign_id_still_raises(self):
        m = _move()
        sorter = _sorter(20)
        self._cache(m, sorter)
        with self.assertRaises(RuntimeError):
            m._fstat_ctr_lookup(
                np.array([3]), model=object(), band_sorter=sorter)
        # beyond-the-last-id misses are foreign too, not an index error
        with self.assertRaises(RuntimeError):
            m._fstat_ctr_lookup(
                np.array([19]), model=object(), band_sorter=sorter)

    def test_legacy_cache_without_unit_ids_raises_on_miss(self):
        m = _move()
        m._fstat_ctr = {"ids": np.array([3, 7, 11])}
        with self.assertRaises(RuntimeError):
            m._fstat_ctr_lookup(np.array([7, 8]))

    def test_in_unit_miss_without_model_raises(self):
        m = _move()
        sorter = _sorter(20)
        self._cache(m, sorter)
        with self.assertRaises(RuntimeError):
            m._fstat_ctr_lookup(np.array([7]))  # reserve row, no fallback args


class SnrTruncAlphaTest(unittest.TestCase):
    def test_analytic_boundary(self):
        m = _move()
        # snr_center 50, limit 5 -> ln(10)/sigma
        alpha = m._snr_trunc_alpha(
            np.log(np.array([50.0])), np.array([0.5]), 5.0)
        np.testing.assert_allclose(alpha, np.log(10.0) / 0.5)

    def test_degenerate_clamps_to_floor(self):
        m = _move()
        alpha = m._snr_trunc_alpha(
            np.log(np.array([0.1])), np.array([0.2]), 5.0)
        np.testing.assert_allclose(alpha, m._SNR_TRUNC_ALPHA_FLOOR)
        # normalization stays finite at the floor
        self.assertTrue(np.isfinite(np.log(m._std_norm_cdf(alpha))).all())


class TruncatedDrawTest(unittest.TestCase):
    def test_distance_basis_upper_boundary_respected(self):
        m = _move(basis="dist")
        _cp.random.seed(42)
        alpha = np.full(4000, 0.5)
        z = np.asarray(m._truncnorm_std_draw(4000, alpha))
        self.assertLessEqual(float(z.max()), 0.5 + 1e-9)

    def test_amplitude_basis_lower_boundary_respected(self):
        m = _move(basis="lnA")
        _cp.random.seed(42)
        alpha = np.full(4000, 0.5)
        z = np.asarray(m._truncnorm_std_draw(4000, alpha))
        self.assertGreaterEqual(float(z.min()), -0.5 - 1e-9)

    def test_alpha_large_reduces_to_standard_normal(self):
        m = _move(basis="dist")
        _cp.random.seed(7)
        z = np.asarray(m._truncnorm_std_draw(4000, np.full(4000, 12.0)))
        self.assertLess(abs(float(z.mean())), 0.1)
        self.assertLess(abs(float(z.std()) - 1.0), 0.1)

    def test_one_uniform_per_row(self):
        # the draw must consume EXACTLY n uniforms from the module stream
        m = _move(basis="dist")
        alpha = np.full(16, 1.25)
        _cp.random.seed(123)
        z = np.asarray(m._truncnorm_std_draw(16, alpha))
        _cp.random.seed(123)
        u = np.asarray(_cp.random.rand(16))
        p = np.clip(u * _phi(alpha), 1e-15, 1.0 - 1e-16)
        np.testing.assert_allclose(
            z, np.sqrt(2.0) * erfinv(2.0 * p - 1.0))


class TruncatedDensityTest(unittest.TestCase):
    def test_normalizes_to_one_distance_basis(self):
        m = _move(basis="dist")
        ln_center, sigma, alpha = 0.0, 0.5, 1.0
        lv = np.linspace(ln_center - 9 * sigma, ln_center + sigma * alpha, 40001)
        v = np.exp(lv)
        g = np.exp(m._slot0_log_proposal(
            v, np.full_like(v, ln_center), np.full_like(v, sigma),
            alpha=np.full_like(v, alpha)))
        self.assertAlmostEqual(_trapz(g, v), 1.0, places=3)

    def test_normalizes_to_one_amplitude_basis(self):
        m = _move(basis="lnA")
        ln_center, sigma, alpha = -50.0, 0.4, 0.8
        lv = np.linspace(ln_center - sigma * alpha, ln_center + 9 * sigma, 40001)
        g = np.exp(m._slot0_log_proposal(
            lv, np.full_like(lv, ln_center), np.full_like(lv, sigma),
            alpha=np.full_like(lv, alpha)))
        self.assertAlmostEqual(_trapz(g, lv), 1.0, places=3)

    def test_alpha_large_equals_untruncated_exactly(self):
        m = _move(basis="dist")
        v = np.exp(np.linspace(-2.0, 2.0, 101))
        c = np.zeros_like(v)
        s = np.full_like(v, 0.7)
        base = m._slot0_log_proposal(v, c, s)
        trunc = m._slot0_log_proposal(v, c, s, alpha=np.full_like(v, 15.0))
        np.testing.assert_array_equal(base, trunc)

    def test_out_of_support_gets_floor(self):
        m = _move(basis="dist")
        # value ABOVE the max-distance boundary -> log-density -1e300
        v = np.array([np.exp(2.0)])  # u = 2/0.5 = 4 > alpha = 1
        out = m._slot0_log_proposal(
            v, np.zeros(1), np.array([0.5]), alpha=np.array([1.0]))
        self.assertEqual(float(out[0]), -1e300)
        # amplitude basis mirrors on the LOWER side
        m2 = _move(basis="lnA")
        out2 = m2._slot0_log_proposal(
            np.array([-2.0]), np.zeros(1), np.array([0.5]),
            alpha=np.array([1.0]))
        self.assertEqual(float(out2[0]), -1e300)

    def test_birth_death_density_symmetry(self):
        # birth-side draws, transformed exactly as _run_rj_step stores them,
        # must be inside the support and score identically when re-evaluated
        # death-side with the same per-row (ln_center, sigma, alpha).
        m = _move(basis="dist")
        np.random.seed(5)
        n = 500
        ln_center = np.random.uniform(-1.0, 1.0, n)
        sigma = np.random.uniform(0.05, 0.6, n)
        alpha = m._snr_trunc_alpha(
            np.log(np.random.uniform(5.5, 200.0, n)), sigma, 5.0)
        z = m._truncnorm_std_draw(n, alpha)
        v = np.exp(ln_center + sigma * z)  # stored slot-0 value
        birth = m._slot0_log_proposal(v, ln_center, sigma, alpha=alpha)
        death = m._slot0_log_proposal(v, ln_center, sigma, alpha=alpha)
        self.assertTrue(np.all(birth > -1e299))  # in-support, finite
        np.testing.assert_array_equal(birth, death)

    def test_degenerate_alpha_density_finite_inside_clamped_support(self):
        m = _move(basis="dist")
        alpha = np.array([m._SNR_TRUNC_ALPHA_FLOOR])
        c, s = np.zeros(1), np.array([0.3])
        inside = np.exp(c + s * (m._SNR_TRUNC_ALPHA_FLOOR - 0.5))
        out = m._slot0_log_proposal(np.array([inside[0]]), c, s, alpha=alpha)
        self.assertTrue(np.isfinite(float(out[0])))
        self.assertGreater(float(out[0]), -1e299)


if __name__ == "__main__":
    unittest.main()
