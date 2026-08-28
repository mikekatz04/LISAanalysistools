"""``GB_TEMPER_COMPACT_ROWS``: inert grid rows leave before chunking.

A grid ROW is one (band, walker-permutation) column of the ladder, and a
swap only ever exchanges two TEMPERATURES OF THE SAME ROW -- so a row
with no source at any temperature can never acquire one.
``GB_TEMPER_SKIP_EMPTY`` already proves that set and uses it to skip slab
traffic, but chunks are still cut from the grid in RAW ORDER, so a
production chunk of ~1200 cells holds only ~44% live rows and still pays
a full bind plus all ``ntemps - 1`` rung iterations. Compaction filters
the unit's grid to its active rows first.

THE EQUIVALENCE CLASS IS NOT BIT-IDENTITY. The per-rung Metropolis draw
``cp.random.uniform(size=paccept.shape)`` is sized by the chunk's row
count, so dropping rows shifts the RNG stream. Retained pairs still draw
iid uniforms, so their decisions are distribution-identical.

THE LADDER, HOWEVER, IS EXACT, and that is what this file pins. An inert
pair scores ``paccept == 0.0``, and ``0.0 > log(u)`` holds
unconditionally, so today every inert row is recorded as an ACCEPTED swap
that moves nothing: ``+1`` to BOTH ``band_swaps_accepted`` and
``band_swaps_proposed`` at every rung. Those counters drive
``_adapt_band_temps``. Dropping the rows silently would move the
temperature ladder of every PARTIALLY occupied band.

Two of the counters are fully DETERMINISTIC, which makes the correction
testable exactly rather than statistically:

* ``band_swaps_proposed`` -- every row proposes at every rung, whatever
  the RNG does. So for the fixture's 4 walkers it must be exactly 4 per
  (interior band, rung), knob on or off.
* ``band_swaps_accepted`` for a FULLY inert band -- every row is
  always-accepted, so it must also be exactly 4.

If the restoration were missing, the two fully-empty sub-bands would
report 0 instead of 4 and these assertions would fail loudly.
"""

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager

import numpy as np


def _have_gbgpu() -> bool:
    try:
        import gbgpu  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


NTEMPS_T = 4
SEED = 17
# Interior bands of the fixture grid (band_edges has 7 entries -> 6 bands,
# bands 1..4 are interior; leaf j lives in band j+1).
INTERIOR = slice(1, 5)
# Bands 3 and 4 hold no source at any temperature (see _occupancy in
# tests/test_temper_skip_empty) -> every one of their rows is inert.
FULLY_INERT_BANDS = (3, 4)


@contextmanager
def _env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_stage(compact: str, seed: int = SEED, pattern: str = "sparse",
               extra_env=None):
    """One ``run_tempering`` pass under ``GB_TEMPER_COMPACT_ROWS=compact``."""
    from tests.test_gbspecial_flow import build_fixture
    from tests.test_temper_skip_empty import _build
    from lisatools.globalfit.moves.gbbands import BandSorter
    from lisatools.globalfit.moves.gbspecialstretch import (
        GBSpecialStretchMove,
        _ProposeTimer,
    )

    fx = build_fixture(seed=SEED)
    branch = _build(fx, ntemps=NTEMPS_T, pattern=pattern)
    nwalkers = branch.shape[1]

    move = GBSpecialStretchMove(
        *fx["move_args"], is_rj_prop=False, name="temper_compact",
        stretch_probability=0.5, **fx["move_kwargs"],
    )
    move.temperature_control = fx["temperature_control"]
    move.ntemps = NTEMPS_T
    move.nwalkers = nwalkers
    move.num_bands = len(np.asarray(fx["band_edges"])) - 1
    move.band_units = 1
    move.time = 1
    move._prop_timer = _ProposeTimer()
    move._prop_buffer_cache = {}

    sorter = BandSorter(
        branch,
        move.band_edges,
        move.band_N_vals,
        force_backend="cpu",
        transform_fn=fx["transform"],
        max_data_store_size=512,
        gb=fx["gb"],
        gb_fd_comp=move.gb_fd_comp,
        waveform_kwargs=fx["waveform_kwargs"],
    )

    betas = 1.0 / 2.0 ** np.arange(NTEMPS_T)
    band_temps = np.tile(betas[None, :], (move.num_bands, 1))
    initial = band_temps.copy()

    env = {"GB_TEMPER_COMPACT_ROWS": compact}
    env.update(extra_env or {})
    with _env(**env):
        np.random.seed(seed)
        ll_change, acc, prop = move.run_tempering(
            fx["model"], None, sorter, band_temps
        )

    return dict(
        ll_change=np.asarray(ll_change).copy(),
        accepted=np.asarray(acc).copy(),
        proposed=np.asarray(prop).copy(),
        band_temps=np.asarray(band_temps).copy(),
        initial_temps=initial,
        alive=np.asarray(sorter.inds).copy(),
        residual=np.asarray(fx["acs"].data_shaped[0]).copy(),
        counts=dict(move._prop_timer.counts),
        nwalkers=nwalkers,
    )


@unittest.skipUnless(_have_gbgpu(), "requires gbgpu")
class CompactRowsLadderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.on = _run_stage("1")
        cls.off = _run_stage("0")

    # ---- the saving actually happened (else everything below is vacuous)

    def test_rows_were_really_dropped(self):
        on, off = self.on["counts"], self.off["counts"]
        self.assertLess(
            on["temper_cells"], off["temper_cells"],
            "GB_TEMPER_COMPACT_ROWS=1 scheduled just as many cells -- no "
            "row was compacted out, so the ladder assertions below would "
            "pass trivially",
        )
        self.assertLess(on["temper_pairs"], off["temper_pairs"])
        # what survives is all live: every scheduled row does real work
        self.assertEqual(
            on["temper_cells_filled"], on["temper_cells"],
            "a compacted chunk still contained an unfillable row",
        )

    # ---- the exact ladder restoration

    def test_proposed_counters_are_exactly_restored(self):
        """Deterministic: every row proposes at every rung, always."""
        nw = self.off["nwalkers"]
        for res, label in ((self.on, "ON"), (self.off, "OFF")):
            prop = res["proposed"]
            np.testing.assert_array_equal(
                prop[INTERIOR], np.full_like(prop[INTERIOR], nw),
                err_msg=(
                    f"[{label}] every interior band must record exactly "
                    f"{nw} proposed swaps per rung; with compaction on, a "
                    f"shortfall means the inert-row restoration is missing"
                ),
            )
        np.testing.assert_array_equal(
            self.on["proposed"], self.off["proposed"],
            err_msg="compaction changed band_swaps_proposed",
        )

    def test_accepted_counters_exact_for_fully_inert_bands(self):
        """A fully inert band is always-accepted -> deterministic."""
        nw = self.off["nwalkers"]
        for b in FULLY_INERT_BANDS:
            np.testing.assert_array_equal(
                self.on["accepted"][b],
                np.full_like(self.on["accepted"][b], nw),
                err_msg=(
                    f"band {b} is sourceless everywhere: all {nw} of its "
                    f"rows are inert and always-accepted, so its accepted "
                    f"counter must be exactly {nw} at every rung"
                ),
            )
            np.testing.assert_array_equal(
                self.on["accepted"][b], self.off["accepted"][b],
                err_msg=f"compaction changed accepted counts for band {b}",
            )

    def test_edge_bands_never_participate(self):
        for res in (self.on, self.off):
            self.assertEqual(int(res["proposed"][0].sum()), 0)
            self.assertEqual(int(res["proposed"][-1].sum()), 0)

    def test_compaction_does_not_change_the_ll_ledger(self):
        """Compaction must not move the likelihood ledger AT ALL.

        This replaces an assertion that could not have held (fixed
        2026-08-28). The old version indexed ``ll_change[..., b]`` for b in
        FULLY_INERT_BANDS and required zero. Two things were wrong with it:

        * ``ll_change`` is ``(ntemps, nwalkers)`` -- it has NO band axis, so
          ``[..., 3]`` selected WALKER 3, not band 3, and ``[..., 4]`` was
          plain out of bounds on a size-4 axis. The test never examined a
          band, and would have raised IndexError on its second iteration had
          the first one passed.
        * The nonzero entries it tripped on are present with compaction OFF
          too, byte for byte -- they are ordinary top-rung swap deltas, not
          a miscredit.

        The per-band inertness claim is already covered, correctly and on
        genuinely band-indexed arrays, by the accepted/proposed counter
        tests above. What belongs here is the lever's own contract: it is a
        pure performance change, so the whole ledger must be identical.
        That is strictly stronger than the per-band zero claim it replaces.
        """
        np.testing.assert_array_equal(
            self.on["ll_change"], self.off["ll_change"],
            err_msg="row compaction changed the likelihood ledger",
        )

    def test_alive_mask_and_residual_round_trip(self):
        np.testing.assert_array_equal(
            self.on["alive"], self.off["alive"],
            err_msg="compaction changed the alive-source mask",
        )
        np.testing.assert_array_equal(
            self.on["residual"], self.off["residual"],
            err_msg="the parent residual must return to the same state",
        )

    # ---- sensitivity controls

    def test_the_stage_was_not_degenerate(self):
        acc, prop = self.off["accepted"], self.off["proposed"]
        self.assertGreater(int(prop.sum()), 0)
        self.assertGreater(int(acc.sum()), 0)
        self.assertLess(
            int(acc.sum()), int(prop.sum()),
            "every swap was accepted -- the MH branch was never exercised",
        )

    def test_the_counter_assertions_can_fail(self):
        """Control: the fixture really does contain inert rows.

        If nothing were inert, ``test_proposed_counters_are_exactly_
        restored`` would pass without the restoration code ever running.
        The fill census proves the inert set is non-empty.
        """
        off = self.off["counts"]
        self.assertLess(
            off["temper_cells_filled"], off["temper_cells"],
            "no cell was skippable -- there are no inert rows in this "
            "fixture and the restoration path is never exercised",
        )


if __name__ == "__main__":
    unittest.main()
