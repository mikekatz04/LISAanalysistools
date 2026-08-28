"""Deferred cell-label relabels (``GB_CELL_LABEL_DEFERRED``).

``BandSorter.exchange_cell_labels`` / ``..._batch`` relabel by scanning the
FULL flat source table (``isin`` over 1e6-1e7 rows + a boolean-mask getitem,
which is a forced device sync, + 3 full-table scatters). That runs once per
accepted tempering rung pair (~40k pairs/iteration) and once per vertical
sweep (one sweep per in-model repeat step) -- 30-150 s/row on the
orchestration audit's timing (2026-08-27, candidate 2).

The rework: inside a *window* (a tempering unit, or one in-model repeat
block) SOURCES never change cell -- only cells change labels. So the swaps
accumulate into an O(K) permutation over the window's cells and
:meth:`BandSorter.flush_cell_labels` applies it to the full table in ONE
pass.

The crux is COMPOSITION: cell A swaps with B, then B's *new* label swaps
with C, and the sources that started in A must end up in C. The model is a
SLOT: slot ``j`` is "the sources that held ``uni[j]`` when the window
opened"; ``cur[j]`` is that slot's current label and ``pos[j]`` says which
slot currently holds ``uni[j]`` (events name cells by their CURRENT label,
so they resolve through ``pos``). Sources never leave their slot, which is
exactly why the composition is a permutation and the flush is one gather.

Covered here:

* composition against immediate mode -- singles, batches, and CHAINS that
  touch the same cell across several events (the crux);
* walker-permuting exchanges (the tempering grid permutes walkers, so a
  relabel moves both temp and walker);
* deferral is real: nothing reaches the table until the flush;
* flush idempotence, empty-window no-op, and re-anchoring across a
  non-closing flush (the per-chunk flush point);
* the knob OFF path is byte-identical to today's immediate relabel;
* the standing alarm: ``special_index_check`` holds on FLUSHED state, and
  the sync-free ``n_pending`` guard fires for a consumer that would read
  stale labels;
* consumer visibility through the real chunk-head occupancy census that
  ``run_tempering`` runs at gbspecialstretch.py:9851.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.gbbands import (
    BandSorter,
    pack_special_index,
    unpack_special_index,
)

NWALKERS = 2

ON = {"GB_CELL_LABEL_DEFERRED": "1"}
OFF = {"GB_CELL_LABEL_DEFERRED": "0"}


class _LabelSorter:
    """Duck-typed sorter carrying the REAL BandSorter methods under test.

    Same trick as ``test_vertical_swap._RealMethodSorter``: the relabel
    primitives touch only ``xp``/``special_band_inds``/``temp_inds``/
    ``walker_inds``/``band_inds``/``nwalkers``, so they run unmodified on
    CPU against this stub -- no GPU, no waveform, no branch.
    """

    xp = np

    def __init__(self, t, w, b, nwalkers=NWALKERS):
        self.temp_inds = np.asarray(t).copy()
        self.walker_inds = np.asarray(w).copy()
        self.band_inds = np.asarray(b).copy()
        self.nwalkers = nwalkers
        self.special_band_inds = pack_special_index(
            self.temp_inds, self.walker_inds, self.band_inds, nwalkers
        )
        # every source alive (the census consumer reads this)
        self.inds = np.ones(self.temp_inds.shape[0], dtype=bool)
        self._deferred_labels = None

    # --- the real implementations ---
    get_special_band_index = BandSorter.get_special_band_index
    get_separate_inds_from_special_index = (
        BandSorter.get_separate_inds_from_special_index
    )
    special_index_check = BandSorter.special_index_check
    exchange_cell_labels = BandSorter.exchange_cell_labels
    exchange_cell_labels_batch = BandSorter.exchange_cell_labels_batch
    begin_cell_label_window = BandSorter.begin_cell_label_window
    flush_cell_labels = BandSorter.flush_cell_labels
    _defer_exchange = BandSorter._defer_exchange
    _assert_cell_labels_flushed = BandSorter._assert_cell_labels_flushed

    def arrays(self):
        return (
            self.special_band_inds.copy(),
            self.temp_inds.copy(),
            self.walker_inds.copy(),
            self.band_inds.copy(),
        )


def _table():
    """A table with several sources per cell plus bystanders.

    band 1 / walker 0 holds the 4-rung chain (T0..T3); band 2 carries a
    cross-walker pair (the tempering grid permutes walkers); rows 8-9 are
    bystanders no event ever names.
    """
    #      0  1  2  3  4  5  6  7  8  9
    t = [0, 0, 1, 1, 2, 3, 0, 1, 2, 0]
    w = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    b = [1, 1, 1, 1, 1, 1, 2, 2, 2, 1]
    return _LabelSorter(t, w, b)


def _sp(t, w, b):
    return pack_special_index(
        np.atleast_1d(np.asarray(t)),
        np.atleast_1d(np.asarray(w)),
        np.atleast_1d(np.asarray(b)),
        NWALKERS,
    )


def _events_chain(s):
    """A 3-link CHAIN through band 1 / walker 0 plus a cross-walker batch.

    Written against CURRENT labels, exactly as the production callers do:
    ``run_tempering`` packs the grid's (i1, walker, band) and the vertical
    sweep packs the block-local ``t_i``. Link 2 names the label link 1 just
    created, so a scheme that composed wrongly would move the wrong rows.
    """
    # link 1 (batch, as the vertical sweep calls it): T0 <-> T1
    s.exchange_cell_labels_batch(
        _sp(0, 0, 1), np.array([0]), np.array([0]),
        _sp(1, 0, 1), np.array([1]), np.array([0]),
        bands=np.array([1]),
    )
    # link 2 (single, as run_tempering calls it: scalar temps): T1 <-> T2.
    # T1 is now carried by the rows that started at T0.
    s.exchange_cell_labels(
        _sp(1, 0, 1), 1, np.array([0]),
        _sp(2, 0, 1), 2, np.array([0]),
        bands=np.array([1]),
    )
    # link 3 (batch): T2 <-> T3 -- the same rows move a third time
    s.exchange_cell_labels_batch(
        _sp(2, 0, 1), np.array([2]), np.array([0]),
        _sp(3, 0, 1), np.array([3]), np.array([0]),
        bands=np.array([1]),
    )
    # cross-walker exchange in band 2: (T0,w0) <-> (T1,w1). Both the temp
    # AND the walker label move, which the tempering grid does routinely.
    s.exchange_cell_labels(
        _sp(0, 0, 2), 0, np.array([0]),
        _sp(1, 1, 2), 1, np.array([1]),
        bands=np.array([2]),
    )


def _immediate_reference():
    with mock.patch.dict(os.environ, OFF):
        s = _table()
        _events_chain(s)
    return s


class CompositionTest(unittest.TestCase):
    """Deferred + flush == immediate, including chained relabels."""

    def test_chain_matches_immediate(self):
        ref = _immediate_reference()
        with mock.patch.dict(os.environ, ON):
            s = _table()
            self.assertTrue(s.begin_cell_label_window(s.special_band_inds))
            _events_chain(s)
            self.assertTrue(s.flush_cell_labels(close=True))

        np.testing.assert_array_equal(
            s.special_band_inds, ref.special_band_inds)
        np.testing.assert_array_equal(s.temp_inds, ref.temp_inds)
        np.testing.assert_array_equal(s.walker_inds, ref.walker_inds)
        np.testing.assert_array_equal(s.band_inds, ref.band_inds)

    def test_chain_actually_moved_rows_three_rungs(self):
        """Guard the reference itself: rows 0-1 must END at T3.

        Without this the equivalence test would still pass if BOTH paths
        were broken in the same direction.
        """
        ref = _immediate_reference()
        np.testing.assert_array_equal(ref.temp_inds[:2], np.array([3, 3]))
        # and the bystanders never moved
        self.assertEqual(int(ref.temp_inds[8]), 2)
        self.assertEqual(int(ref.temp_inds[9]), 0)

    def test_cross_walker_labels_follow(self):
        ref = _immediate_reference()
        # row 6 started (T0,w0,b2) and takes (T1,w1,b2)
        self.assertEqual(int(ref.temp_inds[6]), 1)
        self.assertEqual(int(ref.walker_inds[6]), 1)
        # row 7 started (T1,w1,b2) and takes (T0,w0,b2)
        self.assertEqual(int(ref.temp_inds[7]), 0)
        self.assertEqual(int(ref.walker_inds[7]), 0)

    def test_deferral_is_real_nothing_lands_before_flush(self):
        """The whole point: no full-table write until the flush."""
        with mock.patch.dict(os.environ, ON):
            s = _table()
            before = s.arrays()
            s.begin_cell_label_window(s.special_band_inds)
            _events_chain(s)
            for got, want in zip(s.arrays(), before):
                np.testing.assert_array_equal(got, want)
            s.flush_cell_labels(close=True)
        self.assertFalse(np.array_equal(s.arrays()[1], before[1]))

    def test_single_event_matches_immediate(self):
        with mock.patch.dict(os.environ, OFF):
            ref = _table()
            ref.exchange_cell_labels(
                _sp(0, 0, 1), 0, np.array([0]),
                _sp(1, 0, 1), 1, np.array([0]), bands=np.array([1]))
        with mock.patch.dict(os.environ, ON):
            s = _table()
            s.begin_cell_label_window(s.special_band_inds)
            s.exchange_cell_labels(
                _sp(0, 0, 1), 0, np.array([0]),
                _sp(1, 0, 1), 1, np.array([0]), bands=np.array([1]))
            s.flush_cell_labels(close=True)
        np.testing.assert_array_equal(s.temp_inds, ref.temp_inds)
        np.testing.assert_array_equal(
            s.special_band_inds, ref.special_band_inds)

    def test_swap_back_is_identity(self):
        """A <-> B then B <-> A returns every row to its original label."""
        with mock.patch.dict(os.environ, ON):
            s = _table()
            before = s.arrays()
            s.begin_cell_label_window(s.special_band_inds)
            s.exchange_cell_labels_batch(
                _sp(0, 0, 1), np.array([0]), np.array([0]),
                _sp(1, 0, 1), np.array([1]), np.array([0]))
            s.exchange_cell_labels_batch(
                _sp(0, 0, 1), np.array([0]), np.array([0]),
                _sp(1, 0, 1), np.array([1]), np.array([0]))
            s.flush_cell_labels(close=True)
        for got, want in zip(s.arrays(), before):
            np.testing.assert_array_equal(got, want)

    def test_multi_pair_batch_matches_immediate(self):
        """K disjoint pairs in ONE batch == K sequential singles."""
        sp_a = np.concatenate([_sp(0, 0, 1), _sp(2, 0, 1)])
        sp_b = np.concatenate([_sp(1, 0, 1), _sp(3, 0, 1)])
        t_a, t_b = np.array([0, 2]), np.array([1, 3])
        w = np.array([0, 0])
        with mock.patch.dict(os.environ, OFF):
            ref = _table()
            ref.exchange_cell_labels_batch(sp_a, t_a, w, sp_b, t_b, w)
        with mock.patch.dict(os.environ, ON):
            s = _table()
            s.begin_cell_label_window(s.special_band_inds)
            s.exchange_cell_labels_batch(sp_a, t_a, w, sp_b, t_b, w)
            s.flush_cell_labels(close=True)
        np.testing.assert_array_equal(s.temp_inds, ref.temp_inds)
        np.testing.assert_array_equal(
            s.special_band_inds, ref.special_band_inds)


class FlushSemanticsTest(unittest.TestCase):

    def test_empty_window_is_a_noop(self):
        with mock.patch.dict(os.environ, ON):
            s = _table()
            before = s.arrays()
            s.begin_cell_label_window(s.special_band_inds)
            self.assertTrue(s.flush_cell_labels(close=True))
        for got, want in zip(s.arrays(), before):
            np.testing.assert_array_equal(got, want)

    def test_flush_is_idempotent(self):
        with mock.patch.dict(os.environ, ON):
            s = _table()
            s.begin_cell_label_window(s.special_band_inds)
            _events_chain(s)
            s.flush_cell_labels(close=True)
            after = s.arrays()
            # window closed: a second flush is a no-op and reports it
            self.assertFalse(s.flush_cell_labels(close=True))
        for got, want in zip(s.arrays(), after):
            np.testing.assert_array_equal(got, want)

    def test_non_closing_flush_reanchors(self):
        """The per-chunk flush point: apply, keep the window open, and go on.

        Splitting the chain across a non-closing flush must land exactly
        where the un-split chain does -- that is what makes the tempering
        chunk boundary a legal flush point.
        """
        ref = _immediate_reference()
        with mock.patch.dict(os.environ, ON):
            s = _table()
            s.begin_cell_label_window(s.special_band_inds)
            # first two links, then flush WITHOUT closing
            s.exchange_cell_labels_batch(
                _sp(0, 0, 1), np.array([0]), np.array([0]),
                _sp(1, 0, 1), np.array([1]), np.array([0]),
                bands=np.array([1]))
            s.exchange_cell_labels(
                _sp(1, 0, 1), 1, np.array([0]),
                _sp(2, 0, 1), 2, np.array([0]), bands=np.array([1]))
            self.assertTrue(s.flush_cell_labels())
            # ... the rest of the chain continues in the SAME window
            s.exchange_cell_labels_batch(
                _sp(2, 0, 1), np.array([2]), np.array([0]),
                _sp(3, 0, 1), np.array([3]), np.array([0]),
                bands=np.array([1]))
            s.exchange_cell_labels(
                _sp(0, 0, 2), 0, np.array([0]),
                _sp(1, 1, 2), 1, np.array([1]), bands=np.array([2]))
            s.flush_cell_labels(close=True)
        np.testing.assert_array_equal(s.temp_inds, ref.temp_inds)
        np.testing.assert_array_equal(s.walker_inds, ref.walker_inds)
        np.testing.assert_array_equal(
            s.special_band_inds, ref.special_band_inds)

    def test_flush_without_a_window_is_false(self):
        with mock.patch.dict(os.environ, ON):
            s = _table()
            self.assertFalse(s.flush_cell_labels())


class KnobOffTest(unittest.TestCase):
    """OFF (the default) is today's immediate relabel, untouched."""

    def test_default_is_off(self):
        env = dict(os.environ)
        env.pop("GB_CELL_LABEL_DEFERRED", None)
        with mock.patch.dict(os.environ, env, clear=True):
            s = _table()
            self.assertFalse(s.begin_cell_label_window(s.special_band_inds))
            self.assertIsNone(s._deferred_labels)

    def test_off_relabels_immediately(self):
        with mock.patch.dict(os.environ, OFF):
            s = _table()
            s.begin_cell_label_window(s.special_band_inds)
            s.exchange_cell_labels_batch(
                _sp(0, 0, 1), np.array([0]), np.array([0]),
                _sp(1, 0, 1), np.array([1]), np.array([0]))
            # landed with no flush at all
            np.testing.assert_array_equal(
                s.temp_inds[:4], np.array([1, 1, 0, 0]))
            self.assertFalse(s.flush_cell_labels(close=True))

    def test_on_and_off_agree_on_the_full_chain(self):
        ref = _immediate_reference()
        with mock.patch.dict(os.environ, ON):
            s = _table()
            s.begin_cell_label_window(s.special_band_inds)
            _events_chain(s)
            s.flush_cell_labels(close=True)
        np.testing.assert_array_equal(s.temp_inds, ref.temp_inds)
        np.testing.assert_array_equal(s.walker_inds, ref.walker_inds)
        np.testing.assert_array_equal(
            s.special_band_inds, ref.special_band_inds)


class StandingAlarmTest(unittest.TestCase):
    """The alarms must keep their meaning under deferral."""

    def test_special_index_check_holds_on_flushed_state(self):
        with mock.patch.dict(os.environ, ON):
            s = _table()
            s.begin_cell_label_window(s.special_band_inds)
            _events_chain(s)
            s.flush_cell_labels(close=True)
            self.assertTrue(bool(s.special_index_check))
            # and the packed key really does decompose to the components
            t, w, b = unpack_special_index(s.special_band_inds, NWALKERS)
            np.testing.assert_array_equal(t, s.temp_inds)
            np.testing.assert_array_equal(w, s.walker_inds)
            np.testing.assert_array_equal(b, s.band_inds)

    def test_pending_guard_fires_for_a_stale_reader(self):
        """``_assert_cell_labels_flushed`` is the sync-free consumer guard.

        Silent until something pends, loud the moment a consumer would read
        a label the deferred table has already moved.
        """
        with mock.patch.dict(os.environ, ON):
            s = _table()
            s.begin_cell_label_window(s.special_band_inds)
            s._assert_cell_labels_flushed("test")  # nothing pending yet
            s.exchange_cell_labels_batch(
                _sp(0, 0, 1), np.array([0]), np.array([0]),
                _sp(1, 0, 1), np.array([1]), np.array([0]))
            with self.assertRaises(AssertionError):
                s._assert_cell_labels_flushed("test")
            s.flush_cell_labels()
            s._assert_cell_labels_flushed("test")  # cleared by the flush

    def test_cell_outside_the_declared_universe_raises_at_flush(self):
        """A window whose universe misses a cell is a programming error.

        Detected device-side with no per-event sync, surfaced at the flush.
        """
        with mock.patch.dict(os.environ, ON):
            s = _table()
            # universe restricted to band 1 -- the band-2 event is outside
            s.begin_cell_label_window(
                s.special_band_inds[s.band_inds == 1])
            s.exchange_cell_labels(
                _sp(0, 0, 2), 0, np.array([0]),
                _sp(1, 1, 2), 1, np.array([1]))
            with self.assertRaises(AssertionError):
                s.flush_cell_labels(close=True)


class ConsumerVisibilityTest(unittest.TestCase):
    """A real consumer path reads correct labels after the flush.

    ``run_tempering`` opens each chunk with the alive-cell occupancy census
    at gbspecialstretch.py:9851 (``main_band_sorter.special_band_inds[
    main_band_sorter.inds]`` -> unique/counts -> searchsorted per grid
    cell). That is THE mid-window consumer the per-chunk flush exists for,
    so it is the one reproduced here.
    """

    @staticmethod
    def _occupancy(sorter, cells):
        alive = sorter.special_band_inds[sorter.inds]
        u_sp, u_ct = np.unique(alive, return_counts=True)
        pos = np.searchsorted(u_sp, cells)
        pos = np.clip(pos, 0, max(u_sp.shape[0] - 1, 0))
        return np.where(u_sp[pos] == cells, u_ct[pos], 0)

    def _cells(self):
        return np.concatenate([_sp(t, 0, 1) for t in range(4)])

    def test_census_matches_immediate_after_flush(self):
        cells = self._cells()
        ref = _immediate_reference()
        want = self._occupancy(ref, cells)

        with mock.patch.dict(os.environ, ON):
            s = _table()
            s.begin_cell_label_window(s.special_band_inds)
            _events_chain(s)
            s.flush_cell_labels(close=True)
        np.testing.assert_array_equal(self._occupancy(s, cells), want)
        # the 3-link chain carried rows 0-1 from T0 all the way to T3,
        # displacing one row down each rung it passed through
        np.testing.assert_array_equal(want, np.array([2, 1, 1, 2]))

    def test_census_is_stale_before_the_flush(self):
        """Why the flush point is load-bearing, stated as a test."""
        cells = self._cells()
        with mock.patch.dict(os.environ, ON):
            s = _table()
            s.begin_cell_label_window(s.special_band_inds)
            _events_chain(s)
            stale = self._occupancy(s, cells)
            s.flush_cell_labels(close=True)
            fresh = self._occupancy(s, cells)
        self.assertFalse(np.array_equal(stale, fresh))


if __name__ == "__main__":
    unittest.main()
