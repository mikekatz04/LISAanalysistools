"""GB parallel residual search on a real WDM residual (CPU, ~3 min).

The synthetic world of ``tests/_para_search_world.py``: 7.6 days (Nf 256 x
Nt 256 x 10 s), a scirdv1 XYZ inverse covariance, NOISE-FREE injections at
optimal SNR 4, 6, 9, 15 in four interior sub-bands of 64 bins plus one band
holding TWO sources (SNR 7 and 6, 8 bins apart) and three empty bands.
Noise-free makes every expectation exact: the F-stat maximum of a source is
its SNR^2 / 2, so the SNR-4 source (F = 8) sits below the SNR-5 gate (F =
12.5) by construction.

Pinned: the search finds every SNR >= 6 source and not the SNR-4 one, splits
the two-source band into two finds, converges and stops every band inside
[min_steps, max_steps], the SNR gate drops what it should, the stage-2
samples sit at the stage-1 maximum (the F-stat maximiser -> template mapping
of docs/gb-para-search.md section 9), the sig-het F-stat agrees with the
exact chunked one, and the epoch npz loads through
``WarmStartComponents.from_npz`` and draws on the found sources.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _para_search_world import build_world, make_builder  # noqa: E402

from lisatools.globalfit.warmstart import para_search as ps  # noqa: E402


def _cfg(**kw):
    base = dict(ntemps=4, nwalkers=16, min_steps=30, converge_window=20,
                max_steps=120, stage2_steps=20, stage2_ntemps=2, seed=5)
    base.update(kw)
    return ps.ParaSearchConfig(**base)


class WorldSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = build_world()
        cls.cfg = _cfg()
        cls.b = make_builder(cls.w, cls.cfg)
        cls.f = cls.b.run(walker=0)

    def _match(self, f0_hz, bins=1.0):
        d = np.abs(self.f["f0_hz"] - f0_hz) / self.w["df"]
        return np.nonzero(d <= bins)[0]

    def test_finds_every_snr6_plus_source_and_not_the_snr4_one(self):
        for t in self.w["truth"]:
            m = self._match(t["f0_hz"], 1.0 if t["snr"] >= 6 else 4.0)
            with self.subTest(snr=t["snr"], band=t["band"]):
                if t["snr"] >= 6:
                    self.assertEqual(len(m), 1)
                    i = int(m[0])
                    self.assertEqual(int(self.f["band"][i]), t["band"])
                    self.assertAlmostEqual(self.f["opt_snr"][i], t["snr"],
                                           delta=0.2 * t["snr"])
                    self.assertGreaterEqual(self.f["det_snr"][i],
                                            self.cfg.det_frac * self.f["opt_snr"][i])
                    # noise-free: the stage-1 maximum is SNR^2 / 2 (a near
                    # band-mate adds its cross term: within 15 %)
                    self.assertAlmostEqual(self.f["F1"][i], 0.5 * t["snr"] ** 2,
                                           delta=0.15 * 0.5 * t["snr"] ** 2)
                else:
                    self.assertEqual(len(m), 0)
        self.assertEqual(int(self.f["n_finds"]), 5)

    def test_the_two_source_band_splits(self):
        bf = dict(zip(self.f["bands"].tolist(), self.f["band_finds"].tolist()))
        self.assertEqual(bf[5], 2)
        self.assertEqual([b for b, n in bf.items() if n >= 2], [5])
        self.assertEqual(bf[3], 0)
        self.assertEqual(bf[7], 0)

    def test_every_band_converges_and_stops(self):
        st = self.f["steps"]
        self.assertTrue(np.all(self.f["converged"]))
        self.assertTrue(np.all(st >= self.cfg.min_steps))
        self.assertTrue(np.all(st <= self.cfg.max_steps))
        self.assertEqual(list(self.f["bands"]), list(range(1, 9)))

    def test_stage2_sits_on_the_stage1_maximum(self):
        """lnL of the best stage-2 sample == F of the stage-1 maximum: the
        F-stat extrinsic maximisers seed stage 2 at the right template (the
        production pin's cos(iota) sign would put it 100+ nats lower)."""
        np.testing.assert_allclose(self.f["lnl2"], self.f["F1"], rtol=0.05, atol=1.0)
        n = np.diff(self.f["offsets"])
        self.assertTrue(np.all(n == (self.cfg.stage2_steps // 2) * self.cfg.nwalkers))

    def test_the_gate(self):
        """snr_min 8 on bands 2 (SNR 6) and 6 (SNR 15): only the loud one."""
        b = make_builder(self.w, _cfg(snr_min=8.0))
        g = b.run(bands=[2, 6], walker=0)
        self.assertEqual(int(g["n_finds"]), 1)
        self.assertEqual(int(g["band"][0]), 6)
        self.assertGreaterEqual(float(g["opt_snr"][0]), 8.0)

    def test_sighet_agrees_with_the_exact_fstat(self):
        from lisatools.sampling.fstat_proposal import compute_fstat

        rows = self.f["best"].copy()
        rows[:, 8] = 0.0
        phys = np.asarray(self.w["transform"].both_transforms(rows))
        sc = self.b.scorer
        self.assertEqual(sc.mode, "sighet")
        F_exact = np.asarray(compute_fstat(*sc.fstat_exact(phys)))
        call = sc.fstat_call((self.w["edges"][1], self.w["edges"][-2]))
        F_sig = np.asarray(compute_fstat(*call(phys)))
        sc.release()
        np.testing.assert_allclose(F_sig, F_exact, rtol=1e-2)

    def test_epoch_npz_draws_on_the_finds(self):
        from lisatools.globalfit.warmstart.proposal import WarmStartComponents

        d = tempfile.mkdtemp(prefix="para_world_")
        self.addCleanup(shutil.rmtree, d, True)
        src = ps.merge_finds([self.f], tobs=self.w["tobs"], radius_bins=4.0)
        man = ps.write_epoch(os.path.join(d, "epoch_0000"), src, [self.f],
                             tobs=self.w["tobs"], transform_fn=self.w["transform"],
                             manifest_extra=dict(epoch=0))
        self.assertEqual(man["n_sources"], 5)
        # loaded the way the move loads it (recipe.build_gb_moves): the run's
        # prior box as the uniform floor (the GMM's member boxes cull the bare
        # mixture's density to -inf outside them), circ_images for PE
        e = self.w["edges"]
        box = ([0.001, e[1] * 1e3, 0.001, 0.0, -1.0, 0.0, 0.0, -1.0, -1.0],
               [1e6, e[-2] * 1e3, 1.0, 2 * np.pi, 1.0, np.pi, 2 * np.pi, 1.0, 1.0])
        c = WarmStartComponents.from_npz(os.path.join(d, "epoch_0000", ps.EPOCH_NPZ),
                                         seed=3, floor_box=box, floor_eps=0.05,
                                         circ_images=3)
        x = np.asarray(c.rvs(size=500))
        f0 = x[:, 1] * 1e-3
        d_bins = np.min(np.abs(f0[:, None] - self.f["f0_hz"][None, :]), axis=1) / self.w["df"]
        self.assertGreater(np.mean(d_bins < 2.0), 0.9)          # the floor takes ~5 %
        # inside the prior box the density is finite everywhere (the floor);
        # a mixture draw outside it (an Mc / ratio tail past the box) has
        # zero prior and is rejected by the RJ move's prior gate anyway
        lo, hi = np.asarray(box[0]), np.asarray(box[1])
        inside = np.all((x >= lo) & (x <= hi), axis=1)
        self.assertGreater(inside.mean(), 0.9)
        self.assertTrue(np.all(np.isfinite(np.asarray(c.logpdf(x[inside])))))


if __name__ == "__main__":
    unittest.main()
