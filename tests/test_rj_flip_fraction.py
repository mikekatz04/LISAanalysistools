import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import (
    GBSpecialStretchMove,
    _resolve_rj_flip_fraction,
)


def _move(fraction):
    m = GBSpecialStretchMove.__new__(GBSpecialStretchMove)
    m.rj_flip_fraction = fraction
    m.use_gpu = False
    m._backend_name = "lisatools_cpu"  # self.xp resolves through the backend
    return m


def _picked(n, num_sources):
    ids = np.random.choice(num_sources, size=n, replace=False)
    return {
        "ids": ids,
        "specials": ids * 7,
        "slot_index": np.arange(n, dtype=np.int32),
        "temp_inds": np.zeros(n, dtype=int),
        "walker_inds": np.arange(n) % 4,
        "band_inds": np.arange(n) % 11,
        "N_vals": np.full(n, 128),
    }


class RJFlipFractionTest(unittest.TestCase):
    def test_fraction_one_is_passthrough(self):
        m = _move(1.0)
        sorter = SimpleNamespace(num_sources=100)
        picked = _picked(30, 100)
        out = m._apply_rj_flip_fraction(sorter, picked)
        self.assertIs(out, picked)
        self.assertFalse(hasattr(sorter, "_rj_flip_allowed"))

    def test_subset_size_and_without_replacement(self):
        m = _move(0.3)
        sorter = SimpleNamespace(num_sources=100)
        m._apply_rj_flip_fraction(sorter, _picked(30, 100))
        allowed = sorter._rj_flip_allowed
        self.assertEqual(allowed.dtype, np.bool_)
        self.assertEqual(int(allowed.sum()), 30)  # round(0.3 * 100), no repeats

    def test_mask_drawn_once_per_sorter(self):
        m = _move(0.5)
        sorter = SimpleNamespace(num_sources=200)
        m._apply_rj_flip_fraction(sorter, _picked(10, 200))
        mask_first = sorter._rj_flip_allowed
        m._apply_rj_flip_fraction(sorter, _picked(10, 200))
        self.assertIs(sorter._rj_flip_allowed, mask_first)
        # a NEW sorter (new proposal) gets its own draw
        sorter2 = SimpleNamespace(num_sources=200)
        m._apply_rj_flip_fraction(sorter2, _picked(10, 200))
        self.assertIsNot(sorter2._rj_flip_allowed, mask_first)

    def test_filter_consistent_across_keys(self):
        np.random.seed(3)
        m = _move(0.4)
        sorter = SimpleNamespace(num_sources=50)
        picked = _picked(50, 50)  # every source picked -> keep == allowed
        out = m._apply_rj_flip_fraction(sorter, picked)
        self.assertIsNotNone(out)
        allowed = sorter._rj_flip_allowed
        self.assertTrue(np.all(allowed[out["ids"]]))
        n_kept = len(out["ids"])
        for key, value in out.items():
            self.assertEqual(len(value), n_kept, key)
        # rows outside the subset never appear
        dropped = picked["ids"][~allowed[picked["ids"]]]
        self.assertFalse(np.isin(out["ids"], dropped).any())

    def test_empty_batch_returns_none(self):
        m = _move(0.02)  # keeps max(1, round(0.02*100)) = 2 of 100
        sorter = SimpleNamespace(num_sources=100)
        m._apply_rj_flip_fraction(sorter, _picked(5, 100))
        allowed = sorter._rj_flip_allowed
        blocked = np.where(~allowed)[0][:5]
        picked = _picked(5, 100)
        picked["ids"] = blocked
        out = m._apply_rj_flip_fraction(sorter, picked)
        self.assertIsNone(out)

    def test_minimum_one_slot(self):
        m = _move(0.001)
        sorter = SimpleNamespace(num_sources=10)  # round(0.001*10) = 0 -> 1
        m._apply_rj_flip_fraction(sorter, _picked(10, 10))
        self.assertEqual(int(sorter._rj_flip_allowed.sum()), 1)


class ResolveRJFlipFractionTest(unittest.TestCase):
    def test_kwarg_wins_over_env(self):
        import os
        os.environ["GB_RJ_FLIP_FRACTION"] = "0.7"
        self.addCleanup(os.environ.pop, "GB_RJ_FLIP_FRACTION", None)
        self.assertEqual(_resolve_rj_flip_fraction("gb", 0.4), 0.4)
        self.assertEqual(_resolve_rj_flip_fraction("gb", None), 0.7)

    def test_default_is_one(self):
        self.assertEqual(_resolve_rj_flip_fraction("gb", None), 1.0)

    def test_bounds(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                _resolve_rj_flip_fraction("gb", bad)

    def test_vgb_has_no_rj_knob_surface(self):
        import os
        # env is IGNORED for the fixed-leaf vgb branch...
        os.environ["VGB_RJ_FLIP_FRACTION"] = "0.3"
        self.addCleanup(os.environ.pop, "VGB_RJ_FLIP_FRACTION", None)
        self.assertEqual(_resolve_rj_flip_fraction("vgb", None), 1.0)
        # ...and an explicit kwarg is rejected, not silently dropped.
        with self.assertRaises(ValueError):
            _resolve_rj_flip_fraction("vgb", 0.5)


if __name__ == "__main__":
    unittest.main()
