"""VGB carries no leaf-cap cell state (user ruling 2026-08-22).

Leaf caps gate RJ births; VGB has no RJ surface. ``leaf_caps=False`` on
``initialize_band_information`` must (a) allocate no cap keys on a fresh
store, (b) skip the cap-grid consistency check on resume, and (c) DROP any
stored cap keys -- so a band-grid migration (e.g. the v6 VGB_BAND_LAYERS
coarsening) never needs a cap-grid companion for vgb, and stale cap
datasets in an old store become inert.
"""

import unittest

import numpy as np

from lisatools.globalfit.state import GBState

NW, NT, NB = 4, 3, 6
EDGES = np.linspace(1e-3, 7e-3, NB + 1)
STALE_CAP = np.linspace(1e-3, 7e-3, 5 * NB + 1)  # old fine cap grid, 30 cells

CAP_KEYS = (
    "cap_edges", "num_cap_cells", "cap_cell_leaf_cap",
    "cap_cell_iters", "cap_cell_best_ll", "cap_cell_cold_ll",
)


def _band_temps():
    return np.tile(np.linspace(1.0, 0.01, NT), (NB, 1))


def _stored_band_info(with_stale_cap):
    """A resume-shaped band_info dict (what GBHDFBackend hands back)."""
    bi = {
        "nwalkers": NW,
        "ntemps": NT,
        "num_bands": NB,
        "band_edges": EDGES.copy(),
        "band_temps": _band_temps(),
        "band_swaps_proposed": np.zeros((NB, NT - 1), dtype=int),
        "band_swaps_accepted": np.zeros((NB, NT - 1), dtype=int),
        "band_num_proposed": np.zeros((NB, NT), dtype=int),
        "band_num_accepted": np.zeros((NB, NT), dtype=int),
        "band_num_proposed_rj": np.zeros((NB, NT), dtype=int),
        "band_num_accepted_rj": np.zeros((NB, NT), dtype=int),
        "band_num_binaries": np.zeros((NT, NW, NB), dtype=int),
    }
    if with_stale_cap:
        bi["cap_edges"] = STALE_CAP.copy()
        bi["num_cap_cells"] = len(STALE_CAP) - 1
    return bi


class FreshNoCapTest(unittest.TestCase):
    def test_fresh_leaf_caps_false_allocates_no_cap_keys(self):
        s = GBState(None)
        nt = s.initialize_band_information(
            NW, NT, EDGES.copy(), _band_temps(), branch_name="vgb",
            leaf_caps=False,
        )
        self.assertEqual(nt, NT)
        for key in CAP_KEYS:
            self.assertNotIn(key, s.band_info)
        # band-level cap family stays (monitor compatibility), sentinel -1
        np.testing.assert_array_equal(
            s.band_info["band_leaf_cap"], np.full(NB, -1)
        )
        # backend statics degrade to the band grid (fallback path)
        np.testing.assert_array_equal(s.static_arrays()["cap_edges"], EDGES)
        self.assertEqual(s.storage_attrs()["num_cap_cells"], NB)

    def test_fresh_leaf_caps_true_unchanged(self):
        s = GBState(None)
        s.initialize_band_information(
            NW, NT, EDGES.copy(), _band_temps(), branch_name="gb",
        )
        self.assertIn("cap_edges", s.band_info)
        np.testing.assert_array_equal(s.band_info["cap_edges"], EDGES)


class ResumeNoCapTest(unittest.TestCase):
    def test_resume_skips_check_and_purges_stale_cap_keys(self):
        s = GBState(None)
        s.band_info = _stored_band_info(with_stale_cap=True)
        # the v6 situation: band grid MATCHES (already migrated), cap grid
        # stale -- must NOT raise, and the stale keys must be gone
        nt = s.initialize_band_information(
            NW, NT, EDGES.copy(), _band_temps(), branch_name="vgb",
            leaf_caps=False,
        )
        self.assertEqual(nt, NT)
        for key in CAP_KEYS:
            self.assertNotIn(key, s.band_info)

    def test_resume_band_grid_check_still_enforced(self):
        s = GBState(None)
        s.band_info = _stored_band_info(with_stale_cap=True)
        other = np.linspace(1e-3, 7e-3, NB + 3)  # different band grid
        with self.assertRaisesRegex(ValueError, "band grid mismatch"):
            s.initialize_band_information(
                NW, NT, other, np.tile(np.linspace(1.0, 0.01, NT), (NB + 2, 1)),
                branch_name="vgb", leaf_caps=False,
            )

    def test_resume_leaf_caps_true_still_refuses_stale_cap(self):
        s = GBState(None)
        s.band_info = _stored_band_info(with_stale_cap=True)
        with self.assertRaisesRegex(ValueError, "leaf-cap grid mismatch"):
            s.initialize_band_information(
                NW, NT, EDGES.copy(), _band_temps(), branch_name="gb",
            )


if __name__ == "__main__":
    unittest.main()
