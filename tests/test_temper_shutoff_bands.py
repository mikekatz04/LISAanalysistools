"""``GB_TEMPER_SKIP_SHUTOFF_BANDS``: shut-off bands take no swaps.

USER RULING 2026-08-28. The high-frequency barren-band shutoff was
REVERSED from "births only" to a FULL FREEZE -- "we want to shutoff that
band for RJ and fancy swaps until it resets". ``run_proposal`` already
drops every row of a shut-off band from the RJ subset; this knob extends
the freeze to the tempering grid, so no cell of a shut-off band is built,
scored or swapped until the band is revived.

COUSIN OF ``GB_TEMPER_COMPACT_ROWS``, NOT A DUPLICATE -- and that is the
central thing this file proves. Compaction drops rows that are inert
because NO temperature holds a source. A shut-off band is frozen even
when hot chains DO hold prior-drawn junk leaves in it, which is exactly
the case compaction keeps. So the band shut off here (band 1) is chosen
to be OCCUPIED at two temperatures across every walker: compaction alone
would schedule all of it.

LADDER SEMANTICS, DELIBERATELY DIFFERENT FROM COMPACTION. Inert rows are
always-accepted, so their counter contribution is deterministic and is
restored exactly. A shut-off band's rows may hold real templates whose
swaps would have been SCORED, so there is no deterministic contribution
to restore: the band simply stops producing swap statistics. Its
``accepted/proposed`` ratio is then 0 at every rung, and
``_adapt_band_temps`` turns an all-equal ratio column into ``dSs == 0``
-- the band's ladder FREEZES while it is shut off. That is the intent,
and it is pinned below. ``_adapt_band_temps`` is per-band (independent
columns), so no OTHER band's ladder may move as a result; that is pinned
too, because it is the failure mode that would corrupt the run.
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
# Band 1 is occupied at temps 0 and 1 for EVERY walker, so no row of it is
# inert -- compaction would keep all of them. Shutting it off is therefore
# a strictly different filter.
SHUT_BAND = 1
OTHER_INTERIOR = (2, 3, 4)


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


def _run_stage(skip_shut: str, shut_bands=(SHUT_BAND,), enabled=True,
               seed: int = SEED):
    """One ``run_tempering`` pass with a hand-set shut-off band set.

    ``enabled`` mimics the production gate ``_band_shutoff_enabled()``,
    which is only true on the F-stat search birth move; the fixture's move
    is not an RJ move, so it is forced here.
    """
    from tests.test_gbspecial_flow import build_fixture
    from tests.test_temper_skip_empty import _build
    from lisatools.globalfit.moves.gbbands import BandSorter
    from lisatools.globalfit.moves.gbspecialstretch import (
        GBSpecialStretchMove,
        _ProposeTimer,
    )

    fx = build_fixture(seed=SEED)
    branch = _build(fx, ntemps=NTEMPS_T, pattern="sparse")
    nwalkers = branch.shape[1]

    move = GBSpecialStretchMove(
        *fx["move_args"], is_rj_prop=False, name="temper_shutoff_search",
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

    if shut_bands is not None:
        shut = np.zeros(move.num_bands, dtype=bool)
        for b in shut_bands:
            shut[b] = True
        move._rj_band_shutoff = shut
    move._band_shutoff_enabled = lambda: bool(enabled)

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

    with _env(GB_TEMPER_SKIP_SHUTOFF_BANDS=skip_shut):
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
class ShutOffBandTemperingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.on = _run_stage("1")
        cls.off = _run_stage("0")

    def test_the_shut_off_band_is_genuinely_occupied(self):
        """Control: this must not be a re-test of inert-row compaction."""
        off = self.off
        self.assertGreater(
            int(off["accepted"][SHUT_BAND].sum()), 0,
            f"band {SHUT_BAND} recorded no accepted swaps -- pick an "
            f"occupied band or this proves nothing beyond compaction",
        )
        self.assertLess(
            int(off["accepted"][SHUT_BAND].sum()),
            int(off["proposed"][SHUT_BAND].sum()),
            f"band {SHUT_BAND} accepted every swap, so it behaves like an "
            f"inert band -- the test would not distinguish the two levers",
        )
        self.assertTrue(
            np.any(off["ll_change"][..., SHUT_BAND] != 0.0),
            f"band {SHUT_BAND} moved no likelihood: it is not occupied",
        )

    def test_shut_off_band_takes_no_swaps(self):
        on = self.on
        self.assertEqual(
            int(on["proposed"][SHUT_BAND].sum()), 0,
            "a shut-off band still proposed swaps",
        )
        self.assertEqual(
            int(on["accepted"][SHUT_BAND].sum()), 0,
            "a shut-off band still recorded accepted swaps -- shut-off "
            "rows must NOT receive the inert-row counter restoration",
        )
        self.assertTrue(
            np.all(on["ll_change"][..., SHUT_BAND] == 0.0),
            "a shut-off band was credited a likelihood change",
        )

    def test_other_bands_are_untouched_in_the_deterministic_counter(self):
        nw = self.off["nwalkers"]
        prop = self.on["proposed"]
        for b in OTHER_INTERIOR:
            np.testing.assert_array_equal(
                prop[b], np.full_like(prop[b], nw),
                err_msg=(
                    f"band {b} is not shut off, so it must still record "
                    f"exactly {nw} proposed swaps per rung"
                ),
            )
        np.testing.assert_array_equal(
            prop[list(OTHER_INTERIOR)],
            self.off["proposed"][list(OTHER_INTERIOR)],
            err_msg="excluding one band changed another band's counters",
        )

    def test_the_shut_off_bands_ladder_freezes(self):
        np.testing.assert_allclose(
            self.on["band_temps"][SHUT_BAND],
            self.on["initial_temps"][SHUT_BAND],
            rtol=1e-12, atol=0.0,
            err_msg=(
                "a shut-off band produces no swap statistics, so its "
                "temperature ladder must not adapt while it is frozen"
            ),
        )

    def test_other_bands_ladders_are_not_corrupted(self):
        """The failure mode that would silently damage a production run."""
        for b in OTHER_INTERIOR:
            np.testing.assert_allclose(
                self.on["band_temps"][b], self.off["band_temps"][b],
                rtol=1e-12, atol=0.0,
                err_msg=(
                    f"shutting off band {SHUT_BAND} moved band {b}'s "
                    f"temperature ladder -- _adapt_band_temps must stay "
                    f"per-band"
                ),
            )

    def test_work_actually_shrank(self):
        self.assertLess(
            self.on["counts"]["temper_cells"],
            self.off["counts"]["temper_cells"],
            "the shut-off band's cells were still scheduled",
        )

    def test_residual_and_alive_mask_round_trip(self):
        np.testing.assert_array_equal(self.on["alive"], self.off["alive"])
        np.testing.assert_array_equal(
            self.on["residual"], self.off["residual"],
            err_msg="the parent residual must return to the same state",
        )


@unittest.skipUnless(_have_gbgpu(), "requires gbgpu")
class ShutOffGatingTest(unittest.TestCase):
    """The exclusion must respect the production gate and the empty set."""

    def test_disabled_move_is_unaffected(self):
        """``_band_shutoff_enabled()`` false -> knob is inert.

        The shutoff set is only ever populated on the F-stat search birth
        move; every other move must temper exactly as before. No rows are
        dropped, so the RNG stream is untouched and this is bit-identical.
        """
        on = _run_stage("1", enabled=False)
        off = _run_stage("0", enabled=False)
        for key in ("accepted", "proposed", "ll_change", "band_temps",
                    "residual", "alive"):
            np.testing.assert_array_equal(
                on[key], off[key],
                err_msg=f"knob changed '{key}' on a non-shutoff move",
            )

    def test_absent_shutoff_attribute_is_safe(self):
        """``_rj_band_shutoff`` is created lazily and may not exist."""
        on = _run_stage("1", shut_bands=None)
        off = _run_stage("0", shut_bands=None)
        for key in ("accepted", "proposed", "ll_change", "band_temps"):
            np.testing.assert_array_equal(
                on[key], off[key],
                err_msg=f"missing _rj_band_shutoff changed '{key}'",
            )


if __name__ == "__main__":
    unittest.main()
