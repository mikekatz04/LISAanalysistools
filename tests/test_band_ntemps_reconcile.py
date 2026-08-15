"""Resume-time reconciliation of the band-temperature ladder (2026-08-15).

``GBState.initialize_band_information`` used to hard-``assert`` that the
configured rung count equalled the stored one. A resumed store written with
a different (often buggier) ladder therefore died with a bare
``AssertionError`` naming neither value -- which is exactly what the VGB
betas bugfix produced (config 12 rungs vs a stored 1-rung ladder).

Policy now: on resume the STORED ladder WINS, loudly (a WARNING naming both
counts, the branch, and the re-rung script), and the resolved rung count is
RETURNED so the caller can size the move's temperature machinery off the
ladder that will actually be used. Genuine corruption -- a store whose own
per-band arrays disagree with each other -- still refuses, with a message.

Light fakes only, in the style of ``test_gb_cap_cell_grid.py``: this is
pure bookkeeping on the band-info dict, so it needs no move, ACA, or
backend.
"""

import unittest

import numpy as np

from lisatools.globalfit.state import GBState, make_cap_edges

# 4 bands, 1 mHz wide, 5 -> 9 mHz (Hz)
BAND_EDGES = np.array([5e-3, 6e-3, 7e-3, 8e-3, 9e-3])
NUM_BANDS = len(BAND_EDGES) - 1
NWALKERS = 2


def _ladder(ntemps):
    """The stock geometric ladder the stock variants build."""
    betas = 1.0 / 1.2 ** np.arange(ntemps)
    if ntemps > 1:
        betas[-1] = 1e-4
    return betas


def _stored_state(ntemps, cap_edges=None):
    """A state standing in for one loaded from an h5 (band info present)."""
    st = GBState(None)
    band_temps = np.tile(_ladder(ntemps), (NUM_BANDS, 1))
    st.initialize_band_information(
        NWALKERS, ntemps, BAND_EDGES, band_temps, cap_edges=cap_edges
    )
    return st


class FreshStartTest(unittest.TestCase):
    def test_fresh_keeps_the_configured_ladder_and_returns_it(self):
        st = GBState(None)
        band_temps = np.tile(_ladder(12), (NUM_BANDS, 1))
        out = st.initialize_band_information(
            NWALKERS, 12, BAND_EDGES, band_temps, branch_name="vgb"
        )
        self.assertEqual(out, 12)
        self.assertEqual(st.band_info["ntemps"], 12)
        self.assertEqual(st.band_info["band_temps"].shape, (NUM_BANDS, 12))
        np.testing.assert_allclose(st.band_info["band_temps"][0], _ladder(12))
        # rung-dimensioned counters follow the configured ladder
        self.assertEqual(
            st.band_info["band_swaps_proposed"].shape, (NUM_BANDS, 11)
        )
        self.assertEqual(
            st.band_info["band_num_binaries"].shape, (12, NWALKERS, NUM_BANDS)
        )


class ResumeMatchingTest(unittest.TestCase):
    def test_config_equals_stored_is_unchanged_and_silent(self):
        st = _stored_state(3)
        before = st.band_info["band_temps"].copy()
        with self.assertLogs("lisatools.globalfit.state", level="WARNING") as cm:
            # nothing should warn; log something so assertLogs has a record
            import logging

            logging.getLogger("lisatools.globalfit.state").warning("sentinel")
            out = st.initialize_band_information(
                NWALKERS, 3, BAND_EDGES,
                np.tile(_ladder(3), (NUM_BANDS, 1)),
                branch_name="gb",
            )
        self.assertEqual(len(cm.output), 1)
        self.assertIn("sentinel", cm.output[0])
        self.assertEqual(out, 3)
        np.testing.assert_array_equal(st.band_info["band_temps"], before)


class ResumeMismatchTest(unittest.TestCase):
    """The user-hit case: stored 1 rung, config 12 (VGB_NTEMPS default)."""

    def test_stored_ladder_wins_with_a_warning(self):
        st = _stored_state(1)
        stored_before = st.band_info["band_temps"].copy()
        with self.assertLogs(
            "lisatools.globalfit.state", level="WARNING"
        ) as cm:
            out = st.initialize_band_information(
                NWALKERS, 12, BAND_EDGES,
                np.tile(_ladder(12), (NUM_BANDS, 1)),
                branch_name="vgb",
            )
        msg = "\n".join(cm.output)
        # the resolution
        self.assertEqual(out, 1)
        self.assertEqual(st.band_info["ntemps"], 1)
        np.testing.assert_array_equal(
            st.band_info["band_temps"], stored_before
        )
        # the warning names both counts, the branch, and the fix path
        self.assertIn("'vgb'", msg)
        self.assertIn("stores 1 rung", msg)
        self.assertIn("builds 12", msg)
        self.assertIn("STORED 1-RUNG LADDER WINS", msg)
        self.assertIn("fix_vgb_band_temps.py", msg)

    def test_shrinking_the_configured_ladder_also_defers_to_the_store(self):
        st = _stored_state(8)
        with self.assertLogs("lisatools.globalfit.state", level="WARNING"):
            out = st.initialize_band_information(
                NWALKERS, 2, BAND_EDGES,
                np.tile(_ladder(2), (NUM_BANDS, 1)),
                branch_name="gb",
            )
        self.assertEqual(out, 8)
        self.assertEqual(st.band_info["band_temps"].shape, (NUM_BANDS, 8))

    def test_no_bare_assertionerror_anymore(self):
        st = _stored_state(1)
        try:
            st.initialize_band_information(
                NWALKERS, 12, BAND_EDGES,
                np.tile(_ladder(12), (NUM_BANDS, 1)),
            )
        except AssertionError as exc:  # pragma: no cover - regression guard
            self.fail(f"resume still raises a bare assert: {exc!r}")


class CorruptStoreTest(unittest.TestCase):
    def test_rung_dimension_disagreeing_with_the_ladder_raises(self):
        st = _stored_state(3)
        # a half-migrated store: band_temps re-rung to 3 but the RJ counter
        # left at the old 5-rung width
        st.band_info["band_num_accepted_rj"] = np.zeros(
            (NUM_BANDS, 5), dtype=int
        )
        with self.assertRaises(ValueError) as ctx:
            st.initialize_band_information(
                NWALKERS, 3, BAND_EDGES,
                np.tile(_ladder(3), (NUM_BANDS, 1)),
                branch_name="vgb",
            )
        msg = str(ctx.exception)
        self.assertIn("corrupted band information", msg)
        self.assertIn("band_num_accepted_rj", msg)
        self.assertIn("'vgb'", msg)
        self.assertNotIsInstance(ctx.exception, AssertionError)

    def test_band_dimension_disagreeing_with_the_grid_raises(self):
        st = _stored_state(3)
        st.band_info["band_temps"] = np.zeros((NUM_BANDS + 2, 3))
        with self.assertRaises(ValueError) as ctx:
            st.initialize_band_information(
                NWALKERS, 3, BAND_EDGES,
                np.tile(_ladder(3), (NUM_BANDS, 1)),
                branch_name="gb",
            )
        self.assertIn("corrupted band information", str(ctx.exception))

    def test_walker_mismatch_raises_informatively(self):
        st = _stored_state(3)
        with self.assertRaises(ValueError) as ctx:
            st.initialize_band_information(
                NWALKERS + 1, 3, BAND_EDGES,
                np.tile(_ladder(3), (NUM_BANDS, 1)),
                branch_name="gb",
            )
        msg = str(ctx.exception)
        self.assertIn("walker-count mismatch", msg)
        self.assertIn("nwalkers=2", msg)
        self.assertIn("nwalkers=3", msg)


class CapGridStillGuardedTest(unittest.TestCase):
    """The rung reconciliation must not weaken the grid guards."""

    def test_cap_grid_mismatch_still_refuses_even_at_a_new_rung_count(self):
        st = _stored_state(1, cap_edges=make_cap_edges(BAND_EDGES, 4))
        with self.assertRaises(ValueError) as ctx:
            st.initialize_band_information(
                NWALKERS, 12, BAND_EDGES,
                np.tile(_ladder(12), (NUM_BANDS, 1)),
                cap_edges=make_cap_edges(BAND_EDGES, 8),
                branch_name="vgb",
            )
        self.assertIn("leaf-cap grid mismatch", str(ctx.exception))


class RecipePropagationTest(unittest.TestCase):
    """Point 2: the resolved ladder must reach the move's sizing.

    ``build_vgb_moves`` needs a full built fit to run, so this exercises the
    exact reconciliation block's arithmetic against a real reconciled state
    rather than re-deriving it: the ladder the move would be sized with is
    the stored one, and ``vgb_info.betas`` follows it.
    """

    def test_resolved_ladder_drives_betas_and_ntemps(self):
        from types import SimpleNamespace

        st = _stored_state(1)
        vgb_info = SimpleNamespace(betas=_ladder(12), ntemps=12)
        ntemps = len(vgb_info.betas)
        band_temps = np.tile(np.asarray(vgb_info.betas), (NUM_BANDS, 1))

        resolved = st.initialize_band_information(
            NWALKERS, ntemps, BAND_EDGES, band_temps, branch_name="vgb"
        )
        if resolved == ntemps:
            st.band_info["band_temps"][:] = band_temps
        else:
            ntemps = resolved
            band_temps = np.asarray(st.band_info["band_temps"], dtype=float)
            vgb_info.betas = band_temps[0].copy()

        self.assertEqual(ntemps, 1)
        self.assertEqual(len(vgb_info.betas), 1)
        # what TemperatureControl / move.accepted would be sized with
        self.assertEqual(np.zeros((ntemps, NWALKERS)).shape, (1, NWALKERS))
        # and the state it acts on agrees
        self.assertEqual(st.band_info["band_temps"].shape, (NUM_BANDS, 1))


class GBRecipePropagationTest(unittest.TestCase):
    """The GB caller needs the SAME reconciliation, for a sharper reason.

    ``build_gb_moves`` follows its ``initialize_band_information`` call with
    an explicit ``band_info["band_temps"][:] = band_temps``. Once the rung
    mismatch stopped raising, that assignment became the new failure point:
    a configured (nbands, 12) ladder written into a stored (nbands, 1) slot
    is a raw numpy broadcast error with no diagnosis. These two tests pin
    both halves -- the hazard is real, and the shipped block defuses it.
    """

    def test_unreconciled_assignment_would_raise_broadcast(self):
        st = _stored_state(1)
        band_temps = np.tile(np.asarray(_ladder(12)), (NUM_BANDS, 1))
        st.initialize_band_information(
            NWALKERS, 12, BAND_EDGES, band_temps, branch_name="gb"
        )
        # This is build_gb_moves WITHOUT the reconciliation block.
        with self.assertRaises(ValueError):
            st.band_info["band_temps"][:] = band_temps

    def test_reconciled_gb_block_assigns_cleanly(self):
        from types import SimpleNamespace

        st = _stored_state(1)
        gb_info = SimpleNamespace(betas=_ladder(12), ntemps=12)
        ntemps = len(gb_info.betas)
        band_temps = np.tile(np.asarray(gb_info.betas), (NUM_BANDS, 1))

        resolved = st.initialize_band_information(
            NWALKERS, ntemps, BAND_EDGES, band_temps, branch_name="gb"
        )
        # verbatim shape of the shipped build_gb_moves block
        if int(resolved) != int(ntemps):
            ntemps = int(resolved)
            band_temps = np.asarray(st.band_info["band_temps"]).copy()
            gb_info.betas = band_temps[0].copy()
        st.band_info["band_temps"][:] = band_temps  # must not raise

        self.assertEqual(ntemps, 1)
        self.assertEqual(len(gb_info.betas), 1)
        self.assertEqual(st.band_info["band_temps"].shape, (NUM_BANDS, 1))


if __name__ == "__main__":
    unittest.main()
