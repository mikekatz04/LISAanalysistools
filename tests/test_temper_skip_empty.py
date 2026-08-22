"""``GB_TEMPER_SKIP_EMPTY``: the band-temper stage skips sourceless cells.

The production v6 grid (1232 sub-bands x 24 temps x 24 walkers) carries
sources in ~10% of its (band, temp, walker) cells, yet the tempering stage
built a twin slab and scored a likelihood for every one of them -- the cost
tracked the TOTAL band count, not the occupancy. The skip removes two kinds
of vacuous work, both exactly (not approximately):

* slabs for grid ROWS -- one (band, walker-permutation) column of the
  ladder -- with no source at any temperature (templates only ever move
  between temperatures OF THE SAME ROW, so such a row can never acquire
  one). This subsumes the fully-empty sub-band: every row of an empty band
  is an empty row.
* likelihood scoring for a swap pair whose two cells are both sourceless at
  the moment it is proposed (occupancy is tracked through accepted swaps,
  because a template rides them down the ladder).

The gate here is bit-identity: with ``GB_TEMPER_SKIP_EMPTY=1`` and ``=0``
the stage must produce the SAME swap decisions, the same counters, the same
credited ll deltas and the same post-swap sorter state -- while filling
strictly fewer slots and scoring strictly fewer pairs.

The occupancy pattern is built by hand on the real CPU FD fixture from
``test_gbspecial_flow`` (real buffer, real gb_fd likelihood engine, no
hand-rolled numerics) at 4 temperatures, and covers, per sub-band:

    band 1 -- occupied at temps 0,1, every walker (a both-occupied pair, a
              one-side-empty pair above it, and a both-empty pair above
              that)
    band 2 -- occupied at temp 0 only, walkers 0,1 (one-side-empty pair at
              the bottom of the ladder)
    band 3 -- EMPTY everywhere (fully-empty sub-band: zero fills)
    band 4 -- EMPTY everywhere

NOTE on grid rows: a row of the swap grid is a (band, row-position) column
whose WALKER is redrawn per temperature (``_permute_walkers_for_swaps``), so
a row is fill-free only when nothing in the band is alive at any of the
walkers it drew. That is why the fill saving concentrates on fully-empty
SUB-BANDS -- exactly the user ruling -- while the per-pair scoring skip is
what tracks cell occupancy.
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


def _occupancy(ntemps, nwalkers, pattern="sparse"):
    """(ntemps, nwalkers, nleaves) alive mask; leaf j lives in band j + 1."""
    if pattern == "full":
        # Every cell occupied: nothing to skip on the fill side, and EVERY
        # pair is live -- so the only thing the skip still changes is that
        # scoring reads the pair's two columns instead of the whole ladder.
        return np.ones((ntemps, nwalkers, 4), dtype=bool)
    inds = np.zeros((ntemps, nwalkers, 4), dtype=bool)
    inds[0:2, :, 0] = True          # band 1: temps 0, 1, every walker
    inds[0, 0:2, 1] = True          # band 2: temp 0, walkers 0-1
    #     band 3 (leaf 2): never alive -- fully-empty sub-band
    #     band 4 (leaf 3): never alive -- fully-empty sub-band
    return inds


def _build(fx, ntemps=NTEMPS_T, pattern="sparse"):
    """A 4-temperature GB branch on the fixture's band grid.

    Every (temp, walker, leaf) cell gets its OWN angles/amplitude (only the
    sub-band is shared), so each cell's template -- and therefore each
    walker's residual -- is distinct and the swaps score a real Metropolis
    ratio instead of a degenerate zero.
    """
    from eryn.state import Branch

    band_edges = np.asarray(fx["band_edges"], dtype=float)
    nwalkers = fx["acs"].nwalkers if hasattr(fx["acs"], "nwalkers") else 4
    rng = np.random.default_rng(SEED)

    nleaves, ndim = 4, 8
    coords = np.zeros((ntemps, nwalkers, nleaves, ndim))
    for j in range(nleaves):
        b = j + 1                                   # interior bands 1..4
        f_lo, f_hi = band_edges[b], band_edges[b + 1]
        for t in range(ntemps):
            for w in range(nwalkers):
                f0 = f_lo + (0.3 + 0.4 * rng.random()) * (f_hi - f_lo)
                coords[t, w, j] = [
                    np.log(rng.uniform(4e-21, 9e-21)),  # lnA
                    f0 * 1e3,                           # f0 [mHz]
                    0.0,                                # fdot
                    rng.uniform(0.0, 2 * np.pi),        # phi0
                    rng.uniform(-1.0, 1.0),             # cos_iota
                    rng.uniform(0.0, np.pi),            # psi
                    rng.uniform(0.0, 2 * np.pi),        # lam
                    rng.uniform(-1.0, 1.0),             # sin_beta
                ]
    inds = _occupancy(ntemps, nwalkers, pattern=pattern)
    return Branch(coords, inds=inds)


def _run_stage(skip: str, ntemps=NTEMPS_T, pattern="sparse"):
    """One full ``run_tempering`` pass under ``GB_TEMPER_SKIP_EMPTY=skip``."""
    from tests.test_gbspecial_flow import build_fixture
    from lisatools.globalfit.moves.gbbands import BandSorter
    from lisatools.globalfit.moves.gbspecialstretch import (
        GBSpecialStretchMove,
        _ProposeTimer,
    )

    fx = build_fixture(seed=SEED)
    branch = _build(fx, ntemps=ntemps, pattern=pattern)
    nwalkers = branch.shape[1]

    move = GBSpecialStretchMove(
        *fx["move_args"], is_rj_prop=False, name="temper_skip",
        stretch_probability=0.5, **fx["move_kwargs"],
    )
    move.temperature_control = fx["temperature_control"]
    move.ntemps = ntemps
    move.nwalkers = nwalkers
    move.num_bands = len(np.asarray(fx["band_edges"])) - 1
    move.band_units = 1           # one pass over every interior band
    move.time = 1                 # exercise the ladder adaptation too
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

    betas = 1.0 / 2.0 ** np.arange(ntemps)
    band_temps = np.tile(betas[None, :], (move.num_bands, 1))

    with _env(GB_TEMPER_SKIP_EMPTY=skip):
        np.random.seed(SEED)      # cp is numpy on CPU: one shared stream
        ll_change, acc, prop = move.run_tempering(
            fx["model"], None, sorter, band_temps
        )

    return dict(
        ll_change=np.asarray(ll_change).copy(),
        accepted=np.asarray(acc).copy(),
        proposed=np.asarray(prop).copy(),
        band_temps=np.asarray(band_temps).copy(),
        special=np.asarray(sorter.special_band_inds).copy(),
        temps=np.asarray(sorter.temp_inds).copy(),
        walkers=np.asarray(sorter.walker_inds).copy(),
        bands=np.asarray(sorter.band_inds).copy(),
        alive=np.asarray(sorter.inds).copy(),
        residual=np.asarray(fx["acs"].data_shaped[0]).copy(),
        counts=dict(move._prop_timer.counts),
    )


@unittest.skipUnless(_have_gbgpu(), "requires gbgpu")
class TemperSkipEmptyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.on = _run_stage("1")
        cls.off = _run_stage("0")

    def test_swap_decisions_and_state_bit_identical(self):
        on, off = self.on, self.off
        np.testing.assert_array_equal(
            on["accepted"], off["accepted"],
            err_msg="accepted-swap counters must not change with the skip",
        )
        np.testing.assert_array_equal(on["proposed"], off["proposed"])
        # credited ll deltas: bit-identical (rtol=0, atol=0)
        np.testing.assert_array_equal(
            on["ll_change"], off["ll_change"],
            err_msg="credited per-walker ll deltas must be bit-identical",
        )
        np.testing.assert_array_equal(on["band_temps"], off["band_temps"])
        for key in ("special", "temps", "walkers", "bands", "alive"):
            np.testing.assert_array_equal(
                on[key], off[key],
                err_msg=f"post-swap sorter '{key}' differs with the skip",
            )
        np.testing.assert_array_equal(
            on["residual"], off["residual"],
            err_msg="parent residual must return to the same state",
        )

    def test_the_stage_actually_swapped_something(self):
        """Guard against a vacuous comparison (all-accept / no-op stage)."""
        acc = self.off["accepted"]
        prop = self.off["proposed"]
        self.assertGreater(int(prop.sum()), 0, "no swaps were proposed")
        self.assertGreater(int(acc.sum()), 0, "no swap was ever accepted")
        self.assertLess(
            int(acc.sum()), int(prop.sum()),
            "every swap was accepted -- the fixture is degenerate and the "
            "bit-identity comparison would not exercise the MH branch",
        )
        # and real (occupied) cells changed temperature label
        self.assertTrue(
            np.any(np.asarray(self.off["ll_change"]) != 0.0),
            "no cell was credited an ll change -- nothing occupied moved",
        )

    def test_fewer_slots_filled_and_pairs_scored(self):
        on, off = self.on["counts"], self.off["counts"]
        # every cell is filled with the skip OFF
        self.assertEqual(off["temper_cells_filled"], off["temper_cells"])
        self.assertEqual(off["temper_pairs_scored"], off["temper_pairs"])
        # ... and strictly fewer with it ON
        self.assertLess(on["temper_cells_filled"], off["temper_cells_filled"])
        self.assertLess(on["temper_pairs_scored"], off["temper_pairs_scored"])

    def test_empty_sub_bands_are_never_filled(self):
        """Bands 3 and 4 hold no source anywhere -> they must cost nothing.

        4 interior bands x 4 row positions = 16 grid rows, ntemps cells
        each. Two of the four bands are entirely sourceless, so at most the
        8 rows of bands 1-2 can ever be filled.
        """
        on, off = self.on["counts"], self.off["counts"]
        self.assertEqual(off["temper_cells"], 16 * NTEMPS_T)
        self.assertEqual(off["temper_cells_filled"], 16 * NTEMPS_T)
        self.assertLessEqual(
            on["temper_cells_filled"], 8 * NTEMPS_T,
            "cells of a fully-empty sub-band were still filled",
        )
        # the occupied bands still get their slabs
        self.assertGreaterEqual(on["temper_cells_filled"], 4 * NTEMPS_T)


@unittest.skipUnless(_have_gbgpu(), "requires gbgpu")
class TemperSkipFullOccupancyTest(unittest.TestCase):
    """At 100% occupancy the skip must be a pure no-op on the results.

    Nothing is skippable here -- every row is filled, every pair is live --
    so the ONLY thing that changes is that a pair is scored on its own two
    ladder columns instead of on all ``ntemps`` of them (the stage's
    "not every likelihood is needed" TODO). That restriction has to be
    exact, which is the strictest bit-identity check in this file.
    """

    @classmethod
    def setUpClass(cls):
        cls.on = _run_stage("1", pattern="full")
        cls.off = _run_stage("0", pattern="full")

    def test_nothing_is_skipped(self):
        for r in (self.on, self.off):
            self.assertEqual(
                r["counts"]["temper_cells_filled"], r["counts"]["temper_cells"]
            )
            self.assertEqual(
                r["counts"]["temper_pairs_scored"], r["counts"]["temper_pairs"]
            )

    def test_bit_identical(self):
        on, off = self.on, self.off
        np.testing.assert_array_equal(on["accepted"], off["accepted"])
        np.testing.assert_array_equal(on["proposed"], off["proposed"])
        np.testing.assert_array_equal(
            on["ll_change"], off["ll_change"],
            err_msg="two-column scoring changed the credited ll deltas",
        )
        np.testing.assert_array_equal(on["band_temps"], off["band_temps"])
        for key in ("special", "temps", "walkers", "bands", "alive"):
            np.testing.assert_array_equal(on[key], off[key], err_msg=key)

    def test_swaps_were_non_trivial(self):
        acc, prop = self.off["accepted"], self.off["proposed"]
        self.assertGreater(int(acc.sum()), 0)
        self.assertLess(int(acc.sum()), int(prop.sum()))


if __name__ == "__main__":
    unittest.main()
