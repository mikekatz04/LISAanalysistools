"""Tests for the warm-start GB RJ-birth proposal (workstream B, B3).

CPU-only and hermetic: a tiny synthetic components npz is built in ``setUp``
with the EXACT writer schema of ``scripts/gb/warmstart_fit_from_store.py``
(means/covs/p/mult/n_members/island_id/f0_window_edges/meta-json). Covers:

* rvs/logpdf mutual consistency (importance identity ``E_q[p_ref/q] ~ 1`` on
  the proposal's own draws -- the gmm.py rvs-covariance-bug regression);
* circular wrap correctness (a mean-near-0 phi0 component draws and scores
  consistently across the 0/2pi seam);
* mixture weights proportional to the inclusion probability p;
* f0-windowed logpdf == full (unwindowed) mixture logpdf to tight tolerance;
* uniform floor behaviour (finite logpdf everywhere in the box; never NaN);
* from_npz round trip + schema validation;
* cross-Tobs candidate-window re-derivation (v1: widths unchanged);
* stage wiring: with GB_WARM_START_COMPONENTS set the gb_search stage lists
  ``rj_warm_search`` IMMEDIATELY BEFORE ``rj_fstat_search``; unset leaves the
  stage lists unchanged (asserted on the constructed recipe spec, no data
  built).
"""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lisatools.sampling.warmstart_proposal import (
    CIRCULAR_COLS,
    COLUMN_NAMES,
    F0_WINDOW_GUARD_NSIGMA,
    WarmStartComponents,
)

TOBS = 7776000.0  # 3 months [s]
NDIM = 9


def _make_cov(sigmas, corr_off=0.15):
    """PD covariance: D @ C @ D with C = (1 - c) I + c J (rank-1 uplift)."""
    d = len(sigmas)
    C = (1.0 - corr_off) * np.eye(d) + corr_off * np.ones((d, d))
    D = np.diag(np.asarray(sigmas, dtype=float))
    return D @ C @ D


def _synthetic_components():
    """4 well-separated components; comps 0/1 share an f0 neighborhood."""
    means = np.array([
        # dist  f0[mHz]  Mc    phi0  cosi  psi   alpha  sind  ratio
        [8.0,  2.0000,  0.60, 3.00,  0.30, 1.20, 4.00,  0.20,  0.05],
        [12.0, 2.0008,  0.50, 2.00, -0.40, 0.80, 1.00, -0.50, -0.10],
        [5.0,  5.0000,  0.70, 0.05,  0.00, 0.50, 5.50,  0.60,  0.20],
        [20.0, 8.0000,  0.40, 5.00,  0.70, 2.50, 0.30, -0.80,  0.00],
    ])
    sig = [0.5, 2.0e-5, 0.01, 0.30, 0.05, 0.15, 0.05, 0.05, 0.02]
    covs = np.stack([_make_cov(sig) for _ in range(len(means))])
    p = np.array([0.6, 0.3, 0.9, 0.15])
    return means, covs, p


def _write_npz(path, means, covs, p, tobs=TOBS):
    """The warmstart_fit_from_store.py writer schema, exactly."""
    n = len(means)
    meta = dict(
        store="synthetic", tobs=tobs, df_mhz=1e3 / tobs, last_k=None,
        column_names=COLUMN_NAMES, f0_units="mHz",
        circular_cols={str(k): v for k, v in CIRCULAR_COLS.items()},
        sample_id_def="stored_iteration_index * nwalkers + walker",
        git_head="test", seed=7,
        pipeline="density-valley + single-linkage + satellite-merge v1",
    )
    np.savez_compressed(
        path, means=means, covs=covs, p=p,
        mult=np.ones(n), n_members=np.full(n, 100, dtype=np.int64),
        island_id=np.arange(n, dtype=np.int64),
        f0_window_edges=np.array([[1.99, 2.01], [4.99, 5.01], [7.99, 8.01]]),
        meta=json.dumps(meta),
    )


def _ref_full_logpdf(x, means, covs, weights):
    """Independent full-mixture reference: EVERY component, minimal-image
    circular displacement, inv/slogdet route (no Cholesky, no windowing)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n, d = x.shape
    K = len(means)
    lp = np.empty((n, K))
    for k in range(K):
        diff = x - means[k][None, :]
        for c, period in CIRCULAR_COLS.items():
            dd = diff[:, c]
            diff[:, c] = dd - period * np.round(dd / period)
        inv = np.linalg.inv(covs[k])
        _, logdet = np.linalg.slogdet(covs[k])
        maha = np.einsum("ni,ij,nj->n", diff, inv, diff)
        lp[:, k] = (np.log(weights[k])
                    - 0.5 * (d * np.log(2.0 * np.pi) + logdet)
                    - 0.5 * maha)
    m = lp.max(axis=1)
    return np.log(np.exp(lp - m[:, None]).sum(axis=1)) + m


class WarmStartComponentsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.npz_path = os.path.join(self._tmp.name, "warmstart_test.npz")
        self.means, self.covs, self.p = _synthetic_components()
        _write_npz(self.npz_path, self.means, self.covs, self.p)

    def tearDown(self):
        self._tmp.cleanup()

    def _load(self, **kwargs):
        kwargs.setdefault("seed", 42)
        return WarmStartComponents.from_npz(self.npz_path, **kwargs)

    # -- schema / round trip -------------------------------------------------
    def test_from_npz_round_trip(self):
        ws = self._load()
        self.assertEqual(ws.n_components, 4)
        self.assertEqual(ws.ndim, NDIM)
        # new_tobs defaults to the npz's own meta tobs
        self.assertEqual(ws.tobs, TOBS)
        self.assertAlmostEqual(ws.df_mhz, 1e3 / TOBS, places=15)
        np.testing.assert_allclose(ws.p, self.p)
        np.testing.assert_allclose(ws.mult, np.ones(4))
        self.assertEqual(ws.meta["pipeline"],
                         "density-valley + single-linkage + satellite-merge v1")
        self.assertEqual(ws.f0_window_edges.shape, (3, 2))
        # circular means canonicalized, non-circular means untouched
        np.testing.assert_allclose(
            ws._tables["means"][:, [0, 1, 2, 4, 7, 8]],
            self.means[:, [0, 1, 2, 4, 7, 8]])

    def test_from_npz_rejects_wrong_schema(self):
        bad = os.path.join(self._tmp.name, "bad.npz")
        np.savez(bad, means=self.means, covs=self.covs, p=self.p)
        with self.assertRaises(ValueError):
            WarmStartComponents.from_npz(bad)

    def test_wide_circular_sigma_warns(self):
        covs = self.covs.copy()
        covs[0, 5, 5] = 0.8 ** 2  # psi sigma 0.8 > pi/6
        wide = os.path.join(self._tmp.name, "wide.npz")
        _write_npz(wide, self.means, covs, self.p)
        with self.assertWarns(RuntimeWarning):
            WarmStartComponents.from_npz(wide)

    # -- weights ~ p ---------------------------------------------------------
    def test_weights_proportional_to_p(self):
        ws = self._load()
        np.testing.assert_allclose(ws.weights, self.p / self.p.sum(),
                                   rtol=1e-14)
        # empirical draw fractions track p (assign by nearest f0; the comps
        # are >= 40 sigma apart in f0)
        draws = ws.rvs(size=40000)
        idx = np.abs(draws[:, 1][:, None]
                     - self.means[None, :, 1]).argmin(axis=1)
        frac = np.bincount(idx, minlength=4) / len(draws)
        np.testing.assert_allclose(frac, self.p / self.p.sum(), atol=0.02)

    def test_p_floor_knob(self):
        ws = self._load(p_floor=0.5)
        w = np.maximum(self.p, 0.5)
        np.testing.assert_allclose(ws.weights, w / w.sum(), rtol=1e-14)

    # -- rvs/logpdf mutual consistency (gmm rvs-covariance-bug regression) ---
    def test_importance_identity(self):
        # E_{x~q}[p_ref(x)/q(x)] = 1 for ANY p_ref supported where q > 0.
        # p_ref = the same mixture with covariances shrunk by 0.5, evaluated
        # by the INDEPENDENT reference (inv/slogdet, all components). A wrong
        # rvs scale (the historical 'multiply by the covariance' bug) shifts
        # this ratio far from 1.
        ws = self._load()
        n = 40000
        draws = ws.rvs(size=n)
        lq = ws.logpdf(draws)
        self.assertTrue(np.all(np.isfinite(lq)))
        lref = _ref_full_logpdf(draws, self.means, 0.5 * self.covs,
                                ws.weights)
        est = np.mean(np.exp(lref - lq))
        self.assertAlmostEqual(est, 1.0, delta=0.06)

    def test_single_component_draw_moments(self):
        # Direct Cholesky regression: sample covariance of one component's
        # draws reproduces the stored covariance (NOT its square).
        single = os.path.join(self._tmp.name, "single.npz")
        _write_npz(single, self.means[:1], self.covs[:1], np.array([1.0]))
        ws = WarmStartComponents.from_npz(single, seed=3)
        draws = ws.rvs(size=60000)
        emp = np.cov(draws.T)
        np.testing.assert_allclose(emp, self.covs[0], rtol=0.12, atol=1e-12)
        # logpdf at the mean equals the analytic Gaussian peak density
        _, logdet = np.linalg.slogdet(self.covs[0])
        expect = -0.5 * (NDIM * np.log(2 * np.pi) + logdet)
        got = float(ws.logpdf(self.means[:1])[0])
        self.assertAlmostEqual(got, expect, places=10)

    # -- circular wrap -------------------------------------------------------
    def test_circular_wrap_draw_and_score(self):
        ws = self._load()
        # component 2 has mean phi0 = 0.05, sigma 0.3: draws must be wrapped
        # into [0, 2pi) with mass on BOTH sides of the seam.
        draws = ws.rvs(size=60000)
        idx = np.abs(draws[:, 1][:, None]
                     - self.means[None, :, 1]).argmin(axis=1)
        phi0 = draws[idx == 2, 3]
        self.assertTrue(np.all((phi0 >= 0.0) & (phi0 < 2 * np.pi)))
        near_lo = np.mean(phi0 < 1.0)
        near_hi = np.mean(phi0 > 2 * np.pi - 1.0)
        self.assertGreater(near_lo, 0.5)      # bulk just above 0
        self.assertGreater(near_hi, 0.05)     # real mass wrapped below 2pi
        # scoring is invariant under adding a period to a circular column
        x = self.means[2:3].copy()
        x_shift = x.copy()
        x_shift[0, 3] += 2 * np.pi
        x_shift[0, 5] -= np.pi
        x_shift[0, 6] += 4 * np.pi
        self.assertAlmostEqual(float(ws.logpdf(x)[0]),
                               float(ws.logpdf(x_shift)[0]), places=12)
        # minimal image: phi0 = 2pi - 0.05 is 0.1 BELOW the mean across the
        # seam; by symmetry it scores like phi0 = 0.15 (0.1 above the mean)
        # up to the correlated off-diagonal terms, and far above the naive
        # unwrapped displacement of 2pi - 0.1.
        x_seam = x.copy()
        x_seam[0, 3] = 2 * np.pi - 0.05
        x_sym = x.copy()
        x_sym[0, 3] = 0.15
        self.assertAlmostEqual(float(ws.logpdf(x_seam)[0]),
                               float(ws.logpdf(x_sym)[0]), delta=0.2)
        self.assertGreater(float(ws.logpdf(x_seam)[0]),
                           float(ws.logpdf(x)[0]) - 5.0)

    # -- f0 windowing --------------------------------------------------------
    def test_windowed_logpdf_matches_full_mixture(self):
        ws = self._load()
        rng = np.random.default_rng(11)
        pts = np.concatenate([
            ws.rvs(size=2000),
            self.means,
            self.means + 0.5 * np.sqrt(
                np.diagonal(self.covs, axis1=1, axis2=2)
            ) * rng.standard_normal((4, NDIM)),
        ])
        got = ws.logpdf(pts)
        ref = _ref_full_logpdf(pts, ws._tables["means"], self.covs,
                               ws.weights)
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)

    def test_overlapping_windows_sum_both_components(self):
        # comps 0 and 1 are 8e-4 mHz apart -- inside each other's candidate
        # window -- so a point between them must include BOTH terms.
        ws = self._load()
        mid = 0.5 * (self.means[0] + self.means[1])[None, :]
        got = float(ws.logpdf(mid)[0])
        ref = float(_ref_full_logpdf(mid, ws._tables["means"], self.covs,
                                     ws.weights)[0])
        self.assertAlmostEqual(got, ref, places=8)

    # -- uniform floor -------------------------------------------------------
    def _floor_box(self):
        lo = [0.001, 1.5, 0.1, 0.0, -1.0, 0.0, 0.0, -1.0, -0.5]
        hi = [30.0, 10.0, 1.0, 2 * np.pi, 1.0, np.pi, 2 * np.pi, 1.0, 0.5]
        return lo, hi

    def test_floor_keeps_logpdf_finite_in_box(self):
        lo, hi = self._floor_box()
        ws = self._load(floor_box=(lo, hi), floor_eps=0.05)
        # far from every component but inside the box: exactly the floor
        far = np.array([[15.0, 3.5, 0.5, 1.0, 0.1, 1.0, 3.0, 0.0, 0.1]])
        expect = np.log(0.05) - np.sum(np.log(np.array(hi) - np.array(lo)))
        self.assertAlmostEqual(float(ws.logpdf(far)[0]), expect, places=10)
        # outside the box: -inf, never NaN
        out = far.copy()
        out[0, 1] = 0.5
        val = ws.logpdf(out)
        self.assertTrue(np.isneginf(val[0]))
        self.assertFalse(np.any(np.isnan(val)))
        # near a component the floor is a (1 - eps) tilt of the mixture
        ws0 = self._load(seed=1)
        at_mean = self.means[:1]
        self.assertAlmostEqual(
            float(ws.logpdf(at_mean)[0]),
            float(np.logaddexp(np.log1p(-0.05) + ws0.logpdf(at_mean)[0],
                               np.log(0.05) + expect - np.log(0.05))),
            places=8,
        )

    def test_floor_rvs_fraction(self):
        lo, hi = self._floor_box()
        ws = self._load(floor_box=(lo, hi), floor_eps=0.2, seed=5)
        draws = ws.rvs(size=30000)
        # floor draws are the ones far (in f0) from every component
        dist_f0 = np.abs(draws[:, 1][:, None]
                         - self.means[None, :, 1]).min(axis=1)
        frac_floor = np.mean(dist_f0 > 0.01)
        self.assertAlmostEqual(frac_floor, 0.2, delta=0.03)
        lq = ws.logpdf(draws)
        self.assertTrue(np.all(np.isfinite(lq)))

    # -- cross-Tobs ----------------------------------------------------------
    def test_cross_tobs_window_rederivation(self):
        ws3 = self._load()
        ws6 = self._load(new_tobs=2 * TOBS)
        self.assertAlmostEqual(ws6.df_mhz, 1e3 / (2 * TOBS), places=15)
        sigma_f0 = np.sqrt(self.covs[:, 1, 1])
        np.testing.assert_allclose(
            ws6.window_halfwidth_mhz,
            ws6.window_df * ws6.df_mhz + F0_WINDOW_GUARD_NSIGMA * sigma_f0,
            rtol=1e-14,
        )
        # windows shrink with the finer df ...
        self.assertTrue(np.all(ws6.window_halfwidth_mhz
                               < ws3.window_halfwidth_mhz))
        # ... but the PROPOSED widths are the stored ones (v1: no Fisher
        # rescale): densities at/near the components are identical.
        pts = np.concatenate([self.means, ws3.rvs(size=500)])
        np.testing.assert_allclose(ws6.logpdf(pts), ws3.logpdf(pts),
                                   rtol=1e-12, atol=1e-12)
        # stored (previous-run) island edges are carried through untouched
        np.testing.assert_allclose(ws6.f0_window_edges, ws3.f0_window_edges)

    # -- misc contract -------------------------------------------------------
    def test_rvs_size_shapes(self):
        ws = self._load()
        self.assertEqual(ws.rvs(size=7).shape, (7, NDIM))
        self.assertEqual(ws.rvs(size=(3, 5)).shape, (3, 5, NDIM))

    def test_pickle_deepcopy(self):
        import copy
        import pickle

        ws = self._load()
        ws2 = pickle.loads(pickle.dumps(copy.deepcopy(ws)))
        pts = self.means
        np.testing.assert_allclose(ws2.logpdf(pts), ws.logpdf(pts))


class WarmStartWiringTest(unittest.TestCase):
    """Stage-list wiring: rj_warm_search IMMEDIATELY BEFORE rj_fstat_search.

    Asserted at the recipe/stage-spec level via run_combined_staged.build_fit
    (GB_ONLY composition -- construction only, no data built, no
    materialization)."""

    SCRIPT = (Path(__file__).resolve().parents[1]
              / "scripts" / "fstat_proposal" / "run_combined_staged.py")

    def setUp(self):
        self._env = os.environ.copy()
        for k in ("GB_WARM_START_COMPONENTS", "COMBINED_SMOKE", "GB_ONLY",
                  "STAGE_SKIP_NOISE", "STAGE_NOISE_ONLY",
                  "STAGE_NOISE_VGB_PE", "TOBS_TARGET"):
            os.environ.pop(k, None)
        os.environ["GB_ONLY"] = "1"
        os.environ["NWALKERS"] = "4"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def _build_fit(self):
        spec = importlib.util.spec_from_file_location(
            "run_combined_staged_for_test", str(self.SCRIPT))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.build_fit()

    def test_stage_order_with_and_without_knob(self):
        # WITHOUT the knob: no rj_warm_search anywhere.
        fit = self._build_fit()
        stages = {st.name: [m.name for m in st.moves]
                  for st in fit.recipe.stages}
        self.assertIn("gb_search", stages)
        baseline = stages["gb_search"]
        self.assertNotIn("rj_warm_search", baseline)
        self.assertIn("rj_fstat_search", baseline)
        baseline_pe = stages["full_pe"]
        self.assertNotIn("rj_warm_search", baseline_pe)

        # WITH the knob: rj_warm_search IMMEDIATELY BEFORE rj_fstat_search
        # in gb_search; full_pe untouched (the ruling names search).
        os.environ["GB_WARM_START_COMPONENTS"] = "/tmp/does_not_matter.npz"
        fit2 = self._build_fit()
        stages2 = {st.name: [m.name for m in st.moves]
                   for st in fit2.recipe.stages}
        armed = stages2["gb_search"]
        self.assertIn("rj_warm_search", armed)
        i_warm = armed.index("rj_warm_search")
        i_fstat = armed.index("rj_fstat_search")
        self.assertEqual(i_fstat, i_warm + 1,
                         f"rj_warm_search must be IMMEDIATELY BEFORE "
                         f"rj_fstat_search; stage is {armed}")
        # removing the warm move recovers the baseline exactly
        self.assertEqual([m for m in armed if m != "rj_warm_search"],
                         baseline)
        self.assertEqual(stages2["full_pe"], baseline_pe)


if __name__ == "__main__":
    unittest.main()
