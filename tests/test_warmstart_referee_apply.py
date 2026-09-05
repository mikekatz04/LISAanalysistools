"""Tests for scripts/gb/warmstart_referee_apply.py — the stage-2.5 step that
turns (fit npz, referee npz) into the REFEREED components npz production arms
via GB_WARM_START_COMPONENTS (docs/6mo-run-prep.md warm-start workstream):

* auto-merge same-island pairs whose centroid cross-match exceeds 0.9
  (moment-matched Gaussian merge, weights ~ p, circular minimal image);
* flag blends (p > 0.5 and referee med_ratio < 0.5); kept by default,
  dropped with ``drop_blends=True``;
* output keeps the exact writer schema so
  ``WarmStartComponents.from_npz`` loads it unchanged.
"""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lisatools.sampling.warmstart_proposal import (
    CIRCULAR_COLS, COLUMN_NAMES, WarmStartComponents,
)

TOBS = 7776000.0

_APPLY = (Path(__file__).resolve().parents[1]
          / "scripts" / "gb" / "warmstart_referee_apply.py")


def _load_apply_module():
    spec = importlib.util.spec_from_file_location(
        "warmstart_referee_apply", _APPLY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_cov(sigmas, corr_off=0.15):
    d = len(sigmas)
    C = (1.0 - corr_off) * np.eye(d) + corr_off * np.ones((d, d))
    D = np.diag(np.asarray(sigmas, dtype=float))
    return D @ C @ D


def _fit_npz(path, means, covs, p, mult=None, island_id=None):
    n = len(means)
    meta = dict(
        store="synthetic", tobs=TOBS, df_mhz=1e3 / TOBS, last_k=None,
        column_names=COLUMN_NAMES, f0_units="mHz",
        circular_cols={str(k): v for k, v in CIRCULAR_COLS.items()},
        sample_id_def="stored_iteration_index * nwalkers + walker",
        git_head="test", seed=7,
        pipeline="density-valley + single-linkage + satellite-merge v1",
    )
    np.savez_compressed(
        path, means=means, covs=covs, p=p,
        mult=np.ones(n) if mult is None else np.asarray(mult, dtype=float),
        n_members=np.full(n, 100, dtype=np.int64),
        island_id=(np.arange(n, dtype=np.int64) if island_id is None
                   else np.asarray(island_id, dtype=np.int64)),
        f0_window_edges=np.array([[1.99, 2.01]]),
        meta=json.dumps(meta),
    )


def _referee_npz(path, n, pairs, cross, med_ratio=None):
    med = np.full(n, np.nan) if med_ratio is None else np.asarray(
        med_ratio, dtype=float)
    refereed = np.flatnonzero(np.isfinite(med))
    np.savez_compressed(
        path,
        referee=refereed.astype(np.int64),
        med_match=np.where(np.isfinite(med), 0.95, np.nan),
        min_match=np.where(np.isfinite(med), 0.5, np.nan),
        med_ratio=med,
        n_sampled=np.where(np.isfinite(med), 8, 0).astype(np.int64),
        pairs=np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
        cross_match=np.asarray(cross, dtype=float),
        assigned_counts=np.full(n, 100, dtype=np.int64),
        meta=json.dumps(dict(source="test")),
    )


class RefereeApplyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.mod = _load_apply_module()

        # comps 0/1: same island, near-identical f0 (the split-artifact
        # scenario); comp 2: independent.
        self.means = np.array([
            # dist  f0[mHz] Mc    phi0  cosi  psi   alpha sind  ratio
            [8.0,  2.0000, 0.60, 3.00,  0.30, 1.20, 4.00, 0.20,  0.05],
            [12.0, 2.0002, 0.50, 3.10, -0.40, 0.80, 4.10, 0.10, -0.10],
            [5.0,  5.0000, 0.70, 0.05,  0.00, 0.50, 5.50, 0.60,  0.20],
        ])
        sig = [0.5, 2.0e-5, 0.01, 0.30, 0.05, 0.15, 0.05, 0.05, 0.02]
        self.covs = np.stack([_make_cov(sig) for _ in self.means])
        self.p = np.array([0.6, 0.3, 0.9])
        self.fit = os.path.join(self.dir, "fit.npz")
        _fit_npz(self.fit, self.means, self.covs, self.p,
                 island_id=[0, 0, 1])

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, pairs, cross, med_ratio=None, **kwargs):
        ref = os.path.join(self.dir, "ref.npz")
        _referee_npz(ref, len(self.means), pairs, cross, med_ratio)
        out = os.path.join(self.dir, "refereed.npz")
        self.mod.apply(self.fit, ref, out, **kwargs)
        return np.load(out, allow_pickle=False)

    def test_merges_high_cross_pair(self):
        out = self._run([[0, 1]], [0.95])
        self.assertEqual(len(out["p"]), 2)
        # moment-matched merge, weights ~ p
        w = self.p[:2] / self.p[:2].sum()
        mu = w[0] * self.means[0] + w[1] * self.means[1]
        j = int(np.argmin(np.abs(out["means"][:, 1] - mu[1])))
        got_mu, got_cov = out["means"][j], out["covs"][j]
        # non-circular columns exact
        for c in [0, 1, 2, 4, 7, 8]:
            self.assertAlmostEqual(got_mu[c], mu[c], places=10)
        d0 = self.means[0] - mu
        d1 = self.means[1] - mu
        cov = (w[0] * (self.covs[0] + np.outer(d0, d0))
               + w[1] * (self.covs[1] + np.outer(d1, d1)))
        np.testing.assert_allclose(
            got_cov[np.ix_([0, 1, 2], [0, 1, 2])],
            cov[np.ix_([0, 1, 2], [0, 1, 2])], rtol=1e-10)
        # p = min(1, p0 + p1); members summed
        self.assertAlmostEqual(out["p"][j], min(1.0, 0.6 + 0.3), places=12)
        self.assertEqual(out["n_members"][j], 200)

    def test_keeps_low_cross_pair(self):
        out = self._run([[0, 1]], [0.60])
        self.assertEqual(len(out["p"]), 3)
        np.testing.assert_allclose(out["means"], self.means)

    def test_blend_flag_and_drop(self):
        med = [np.nan, np.nan, 0.30]          # comp 2: p 0.9, ratio 0.3
        out = self._run([], np.zeros((0,)), med_ratio=med)
        self.assertEqual(len(out["p"]), 3)
        np.testing.assert_array_equal(out["blend"], [False, False, True])
        out2 = self._run([], np.zeros((0,)), med_ratio=med, drop_blends=True)
        self.assertEqual(len(out2["p"]), 2)
        self.assertNotIn(5.0, np.round(out2["means"][:, 1], 3))

    def test_output_loads_in_warmstart_components(self):
        out_path = os.path.join(self.dir, "refereed.npz")
        ref = os.path.join(self.dir, "ref.npz")
        _referee_npz(ref, len(self.means), [[0, 1]], [0.95])
        self.mod.apply(self.fit, ref, out_path)
        ws = WarmStartComponents.from_npz(out_path, new_tobs=2 * TOBS)
        self.assertEqual(ws.n_components, 2)
        lp = ws.logpdf(ws.rvs(64))
        self.assertFalse(np.isnan(lp).any())

    def test_circular_merge_minimal_image(self):
        self.means[0, 3] = 6.20               # phi0 just below 2 pi
        self.means[1, 3] = 0.10               # phi0 just above 0
        self.p[:2] = [0.5, 0.5]
        _fit_npz(self.fit, self.means, self.covs, self.p,
                 island_id=[0, 0, 1])
        out = self._run([[0, 1]], [0.95])
        j = int(np.argmin(np.abs(out["means"][:, 1] - 2.0001)))
        phi = out["means"][j, 3]
        # wrap-aware mean sits near the seam (0 or 2 pi), never near pi
        self.assertLess(min(phi, 2 * np.pi - phi), 0.2)


if __name__ == "__main__":
    unittest.main()
