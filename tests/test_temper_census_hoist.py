"""``GB_TEMPER_CENSUS_HOIST``: per-unit tempering occupancy census.

The tempering chunk loop asks "how many live sources sit in each of THIS
chunk's ~1200 cells" by gathering the WHOLE source table
(``special_band_inds[inds]``, 1e6-1e7 rows on production) and sorting it
(``cp.unique(..., return_counts=True)``) -- once per chunk, ~590 chunks
per move. Only the closing ``searchsorted`` is chunk-sized. The hoist
moves the gather+sort to once per UNIT.

WHY IT IS EXACT, and why the naive argument is not enough. ``inds`` is
genuinely frozen for the whole call (births/deaths live in
``run_proposal``), but ``special_band_inds`` is NOT: every chunk ends in
``flush_cell_labels()``, which relabels the rows of cells that swapped.
That even changes the label MULTISET -- a 3-source cell swapping with an
empty one turns three copies of label A into three copies of label B --
so the census counts really do move mid-loop. The hoist survives because
a swap only ever exchanges two temperatures OF THE SAME ROW, every row
belongs to exactly one chunk, and chunks slice the grid into DISJOINT
cell sets: a flush can only redistribute labels among cells the loop has
already finished with. Every cell a LATER chunk queries is untouched.

The gate is therefore BIT-IDENTITY, and this file is built so that a
regression could actually break it:

* ``GB_TEMPER_PRELOAD_CELLS`` is forced small so the unit is cut into
  MANY chunks. At the production default the whole fixture would be a
  SINGLE chunk and per-chunk vs per-unit would be identical by
  construction -- a vacuous pass.
* the fixture is occupied where swaps get accepted, so flushes really do
  relabel cells between chunks;
* ``cp.unique`` calls are counted, proving the hoist fired at all;
* and a control run with a different seed must DIFFER, proving the
  compared arrays are sensitive rather than degenerate.
"""

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from unittest import mock

import numpy as np


def _have_gbgpu() -> bool:
    try:
        import gbgpu  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


NTEMPS_T = 4
SEED = 17
# Force MANY chunks per unit: 8 cells / 4 temps = 2 grid rows per chunk,
# against 4 walkers x 4 interior bands = 16 rows -> 8 chunks. Without this
# the whole unit is one chunk and the comparison proves nothing.
SMALL_CHUNK = "8"


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


class _CountingCp:
    """Proxy around the module's array library that counts ``unique``."""

    def __init__(self, mod):
        object.__setattr__(self, "_mod", mod)
        object.__setattr__(self, "n_unique", 0)

    def __getattr__(self, key):
        return getattr(object.__getattribute__(self, "_mod"), key)

    def unique(self, *args, **kwargs):
        object.__setattr__(self, "n_unique",
                           object.__getattribute__(self, "n_unique") + 1)
        return object.__getattribute__(self, "_mod").unique(*args, **kwargs)


def _run_stage(hoist: str, seed: int = SEED, pattern: str = "sparse"):
    """One ``run_tempering`` pass under ``GB_TEMPER_CENSUS_HOIST=hoist``."""
    from tests.test_gbspecial_flow import build_fixture
    from tests.test_temper_skip_empty import _build
    from lisatools.globalfit.moves import gbspecialstretch as gss
    from lisatools.globalfit.moves.gbbands import BandSorter
    from lisatools.globalfit.moves.gbspecialstretch import (
        GBSpecialStretchMove,
        _ProposeTimer,
    )

    fx = build_fixture(seed=SEED)
    branch = _build(fx, ntemps=NTEMPS_T, pattern=pattern)
    nwalkers = branch.shape[1]

    move = GBSpecialStretchMove(
        *fx["move_args"], is_rj_prop=False, name="temper_hoist",
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

    counting = _CountingCp(gss.cp)
    with _env(GB_TEMPER_CENSUS_HOIST=hoist,
              GB_TEMPER_PRELOAD_CELLS=SMALL_CHUNK):
        with mock.patch.object(gss, "cp", counting):
            np.random.seed(seed)
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
        n_unique=counting.n_unique,
    )


ARRAY_KEYS = ("ll_change", "accepted", "proposed", "band_temps",
              "special", "temps", "walkers", "bands", "alive", "residual")


@unittest.skipUnless(_have_gbgpu(), "requires gbgpu")
class CensusHoistEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.on = _run_stage("1")
        cls.off = _run_stage("0")

    def test_bit_identical(self):
        for key in ARRAY_KEYS:
            np.testing.assert_array_equal(
                self.on[key], self.off[key],
                err_msg=f"hoisting the census changed '{key}'",
            )

    def test_cell_and_pair_accounting_unchanged(self):
        for key in ("temper_cells", "temper_cells_filled",
                    "temper_pairs", "temper_pairs_scored"):
            self.assertEqual(
                self.on["counts"].get(key), self.off["counts"].get(key),
                f"the hoist must not change '{key}' -- it changes only "
                f"WHERE the census is computed, never what it says",
            )

    def test_the_hoist_actually_fired(self):
        """Strictly fewer full-table sorts, or the knob did nothing."""
        self.assertLess(
            self.on["n_unique"], self.off["n_unique"],
            "GB_TEMPER_CENSUS_HOIST=1 did not reduce the number of "
            "cp.unique calls -- the hoist never took effect",
        )

    def test_the_unit_really_was_cut_into_many_chunks(self):
        """Guard against the vacuous single-chunk configuration."""
        rows = self.off["counts"]["temper_cells"] // NTEMPS_T
        rows_per_chunk = int(SMALL_CHUNK) // NTEMPS_T
        self.assertGreaterEqual(
            rows, 4 * rows_per_chunk,
            "the fixture produced too few grid rows to span several "
            "chunks -- per-chunk and per-unit census would agree trivially",
        )
        # one unit (band_units=1) -> the saving is (chunks - 1) sorts
        self.assertGreaterEqual(
            self.off["n_unique"] - self.on["n_unique"], 3,
            "fewer sorts were removed than there were extra chunks",
        )

    def test_the_stage_was_not_degenerate(self):
        acc, prop = self.off["accepted"], self.off["proposed"]
        self.assertGreater(int(prop.sum()), 0, "no swaps were proposed")
        self.assertGreater(int(acc.sum()), 0, "no swap was ever accepted")
        self.assertLess(
            int(acc.sum()), int(prop.sum()),
            "every swap was accepted -- the MH branch was never exercised",
        )
        self.assertTrue(
            np.any(self.off["ll_change"] != 0.0),
            "nothing occupied moved, so relabels never stressed the census",
        )

    def test_the_comparison_can_detect_a_difference(self):
        """Sensitivity control: a different RNG seed MUST change results.

        Without this, an all-zero or constant comparison array would make
        every assertion above pass for the wrong reason.
        """
        other = _run_stage("0", seed=SEED + 1)
        differs = any(
            not np.array_equal(other[key], self.off[key])
            for key in ARRAY_KEYS
        )
        self.assertTrue(
            differs,
            "changing the RNG seed changed nothing -- the arrays compared "
            "by this suite are insensitive and bit-identity is vacuous",
        )


if __name__ == "__main__":
    unittest.main()
