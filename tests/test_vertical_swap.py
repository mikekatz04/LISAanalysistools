"""Per-repeat VERTICAL band-temperature swaps in the in-model loop.

Fake-based (same style as ``test_inmodel_repeats``): the swap is pure
bookkeeping -- a relabel plus a closed-form acceptance ratio -- so fakes
exercise every branch of it in milliseconds, with no waveform, no buffer
and no GPU.

Why fakes rather than an end-to-end fixture: the CPU flow fixture
(``test_gbspecial_flow.build_fixture``) carries an all-zero data array, so
its templates contribute ~2e-4 to lnL and its swap pairs are almost all
EMPTY cells. An empty-vs-empty pair has ``paccept == 0``, which passes the
Metropolis test unconditionally and moves nothing -- so it inflates the
"accepted swaps" counter while exercising none of the bookkeeping. That was
measured, not assumed (2026-08-18). End-to-end coverage belongs on the GPU
probe, where cells are genuinely occupied.

Covered here:

* pair selection (same walker, same sub-band, adjacent temperatures);
* the closed-form ratio ``(b_cold - b_hot) * (L_hot - L_cold)``;
* the relabel: block ``t_i``/``beta``, sorter labels, per-cell ledgers;
* the hoisted-array staleness hazard -- after a swap the per-half
  ``beta_s`` must still equal ``band_temps[b_i, t_i]``;
* the block-boundary barrier (``special_index_check``);
* default OFF, and no template buffer ever touched.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import (
    GBSpecialStretchMove,
    _resolve_temper_vertical,
)
from lisatools.globalfit.moves.gbbands import (
    pack_special_index,
    unpack_special_index,
)

NTEMPS, NWALKERS, NBANDS = 4, 2, 3


class _FakeSorter:
    """Minimal BandSorter with the real special-index semantics."""

    def __init__(self, temp_inds, walker_inds, band_inds, nwalkers):
        self.temp_inds = np.asarray(temp_inds).copy()
        self.walker_inds = np.asarray(walker_inds).copy()
        self.band_inds = np.asarray(band_inds).copy()
        self.nwalkers = nwalkers
        self.special_band_inds = self.get_special_band_index(
            self.temp_inds, self.walker_inds, self.band_inds
        )
        self.touched_template_buffer = False

    def get_special_band_index(self, t, w, b):
        return pack_special_index(t, w, b, self.nwalkers)

    @property
    def special_index_check(self):
        return np.all(
            self.special_band_inds
            == self.get_special_band_index(
                self.temp_inds, self.walker_inds, self.band_inds
            )
        )

    def exchange_cell_labels(self, sa, ta, wa, sb, tb, wb, bands=None):
        """Real semantics: both membership maps computed before mutation."""
        keep_a = np.isin(self.special_band_inds, np.atleast_1d(sa))
        keep_b = np.isin(self.special_band_inds, np.atleast_1d(sb))
        self.special_band_inds[keep_a] = np.atleast_1d(sb)[0]
        self.temp_inds[keep_a] = tb
        self.special_band_inds[keep_b] = np.atleast_1d(sa)[0]
        self.temp_inds[keep_b] = ta

    def exchange_cell_labels_batch(self, sa, ta, wa, sb, tb, wb,
                                   bands=None):
        """Batch = pairwise loop over this fake's own exchange (the real
        primitive's equivalence is pinned by
        BatchExchangeEquivalenceTest against the REAL BandSorter)."""
        sa, sb = np.atleast_1d(sa), np.atleast_1d(sb)
        ta, tb = np.atleast_1d(ta), np.atleast_1d(tb)
        wa, wb = np.atleast_1d(wa), np.atleast_1d(wb)
        for k in range(sa.size):
            self.exchange_cell_labels(
                sa[k:k + 1], int(ta[k]), wa[k:k + 1],
                sb[k:k + 1], int(tb[k]), wb[k:k + 1],
                bands=None if bands is None
                else np.atleast_1d(bands)[k:k + 1])

    # a vertical swap must never reach for the template twin: the in-model
    # buffer has none (use_template_arr is True only inside run_tempering)
    def swap_template_slots(self, *a, **k):  # pragma: no cover - must not run
        self.touched_template_buffer = True
        raise AssertionError(
            "vertical swap must not touch the template buffer"
        )


class _Move(GBSpecialStretchMove):
    def __init__(self):  # bypass the production ctor
        pass


def _make_move(seed=3):
    m = _Move()
    m._backend_name = "lisatools_cpu"
    m.use_gpu = False
    m.branch_name = "gb"
    m.name = "fake_vert"
    m.ntemps = NTEMPS
    m.nwalkers = NWALKERS
    m.num_bands = NBANDS
    m.temper_vertical = True
    m._temper_rng = np.random.default_rng(seed)
    return m


def _ladder():
    """(num_bands, ntemps) betas, strictly decreasing along the ladder."""
    return np.tile(
        (1.0 / 2.0 ** np.arange(NTEMPS))[None, :], (NBANDS, 1)
    )


def _rows(band=1, walker=0):
    """One picked row per temperature of a single (walker, band) column."""
    t = np.arange(NTEMPS)
    w = np.full(NTEMPS, walker)
    b = np.full(NTEMPS, band)
    return t, w, b


class VerticalKnobTest(unittest.TestCase):
    def test_default_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_TEMPER_VERTICAL", None)
            self.assertFalse(_resolve_temper_vertical("gb", None))

    def test_env_on(self):
        with mock.patch.dict(os.environ, {"GB_TEMPER_VERTICAL": "1"}):
            self.assertTrue(_resolve_temper_vertical("gb", None))

    def test_kwarg_wins_over_env(self):
        with mock.patch.dict(os.environ, {"GB_TEMPER_VERTICAL": "1"}):
            self.assertFalse(_resolve_temper_vertical("gb", False))

    def test_branch_prefix_isolated(self):
        with mock.patch.dict(os.environ, {"GB_TEMPER_VERTICAL": "1"}):
            self.assertFalse(_resolve_temper_vertical("vgb", None))

    def test_bad_value_rejected(self):
        with mock.patch.dict(os.environ, {"GB_TEMPER_VERTICAL": "yes-please"}):
            with self.assertRaises(ValueError):
                _resolve_temper_vertical("gb", None)


class VerticalPairsTest(unittest.TestCase):
    def test_pairs_are_same_walker_same_band_adjacent_temps(self):
        mv = _make_move()
        t, w, b = _rows()
        hot, cold = mv._vertical_pairs(t, w, b)
        self.assertEqual(len(hot), NTEMPS - 1)
        for h, c in zip(hot, cold):
            self.assertEqual(t[h], t[c] + 1)
            self.assertEqual(w[h], w[c])
            self.assertEqual(b[h], b[c])

    def test_no_pair_across_different_walkers(self):
        """Different walkers means different data slabs -- never a pair."""
        mv = _make_move()
        t = np.array([0, 1])
        w = np.array([0, 1])          # <- differs
        b = np.array([1, 1])
        hot, cold = mv._vertical_pairs(t, w, b)
        self.assertEqual(len(hot), 0)

    def test_no_pair_across_different_bands(self):
        mv = _make_move()
        t = np.array([0, 1])
        w = np.array([0, 0])
        b = np.array([1, 2])          # <- differs
        hot, cold = mv._vertical_pairs(t, w, b)
        self.assertEqual(len(hot), 0)

    def test_non_adjacent_temps_are_not_paired(self):
        mv = _make_move()
        t = np.array([0, 2])
        w = np.array([0, 0])
        b = np.array([1, 1])
        hot, cold = mv._vertical_pairs(t, w, b)
        self.assertEqual(len(hot), 0)


class _SweepFixture:
    """A single (walker, band) column with one picked row per temperature."""

    def __init__(self, ll, seed=3):
        self.mv = _make_move(seed)
        self.t, self.w, self.b = _rows()
        self.slots = np.arange(NTEMPS, dtype=int)
        self.band_temps = _ladder()
        self.beta = self.band_temps[self.b, self.t].copy()
        self.ll_ref = np.asarray(ll, dtype=float)
        self.sorter = _FakeSorter(self.t, self.w, self.b, NWALKERS)
        self.ll_change = np.zeros((NTEMPS, NWALKERS, NBANDS))
        self.prop = np.zeros((2, NTEMPS, NWALKERS, NBANDS), dtype=int)
        self.acc = np.zeros_like(self.prop)
        self.cell_ll = {
            "spec": self.sorter.special_band_inds.copy(),
            "ll0": self.ll_ref.copy(),
            "led0": np.zeros(NTEMPS),
            "rep0": np.zeros(NTEMPS, dtype=int),
        }

    def sweep(self, parity=0):
        return self.mv._vertical_swap_sweep(
            self.sorter, self.band_temps, self.t, self.w, self.b,
            self.slots, self.beta, self.ll_ref, self.ll_change,
            self.prop, self.acc, self.cell_ll, parity,
        )


class VerticalSweepTest(unittest.TestCase):
    def test_favourable_swap_is_always_accepted(self):
        """A hotter rung holding the BETTER model always swaps down.

        paccept = (b_cold - b_hot) * (L_hot - L_cold) > 0 when the hot rung
        has the higher likelihood and b_cold > b_hot, and any positive
        exponent beats log(u) for u in (0, 1).
        """
        # temps 0..3, betas 1, .5, .25, .125; give t=1 a much better ll
        fx = _SweepFixture(ll=[-100.0, -10.0, -100.0, -100.0])
        n = fx.sweep(parity=0)          # pairs with cold rung even: (1,0),(3,2)
        self.assertGreaterEqual(n, 1)
        # the good model moved DOWN to the cold rung
        self.assertEqual(fx.t[1], 0)
        self.assertEqual(fx.t[0], 1)
        self.assertAlmostEqual(fx.beta[1], fx.band_temps[fx.b[1], 0])
        self.assertAlmostEqual(fx.beta[0], fx.band_temps[fx.b[0], 1])

    def test_strongly_unfavourable_swap_is_rejected(self):
        """Cold rung already far better -> exponent very negative."""
        fx = _SweepFixture(ll=[-10.0, -1e6, -10.0, -1e6])
        n = fx.sweep(parity=0)
        self.assertEqual(n, 0)
        np.testing.assert_array_equal(fx.t, np.arange(NTEMPS))

    def test_no_likelihood_and_no_template_buffer_touched(self):
        fx = _SweepFixture(ll=[-100.0, -10.0, -100.0, -100.0])
        ll_before = fx.ll_ref.copy()
        fx.sweep(parity=0)
        # the ratio is closed form: lls are READ, never recomputed
        np.testing.assert_array_equal(fx.ll_ref, ll_before)
        self.assertFalse(fx.sorter.touched_template_buffer)

    def test_sorter_labels_stay_self_consistent(self):
        """The block-boundary barrier's invariant."""
        fx = _SweepFixture(ll=[-100.0, -10.0, -100.0, -100.0])
        fx.sweep(parity=0)
        self.assertTrue(bool(fx.sorter.special_index_check))
        t_unpacked, _, _ = unpack_special_index(
            fx.sorter.special_band_inds, NWALKERS
        )
        np.testing.assert_array_equal(t_unpacked, fx.sorter.temp_inds)

    def test_sorter_sources_actually_change_temperature(self):
        """The SORTER must move, not merely stay self-consistent.

        Self-consistency alone is satisfied trivially by doing nothing --
        that exact gap let a `run_tempering` mutation (deleting the
        ``exchange_cell_labels`` call) survive an earlier version of this
        suite. Assert the observable change itself.
        """
        fx = _SweepFixture(ll=[-100.0, -10.0, -100.0, -100.0])
        before = fx.sorter.temp_inds.copy()
        n = fx.sweep(parity=0)
        self.assertGreaterEqual(n, 1)
        after = fx.sorter.temp_inds
        self.assertTrue(
            bool((before != after).any()),
            "accepted vertical swaps did not relabel a single source in the "
            "sorter -- exchange_cell_labels never took effect",
        )
        # rows 0 and 1 are the accepted pair: their temps must have traded
        self.assertEqual(int(after[0]), int(before[1]))
        self.assertEqual(int(after[1]), int(before[0]))
        # and the count of sources per rung is conserved by a swap
        np.testing.assert_array_equal(
            np.bincount(before, minlength=NTEMPS),
            np.bincount(after, minlength=NTEMPS),
        )

    def test_per_cell_ledgers_follow_the_model(self):
        """ll_change_log / counters are keyed by cell and must trade too."""
        fx = _SweepFixture(ll=[-100.0, -10.0, -100.0, -100.0])
        w0, b0 = fx.w[0], fx.b[0]
        fx.ll_change[0, w0, b0] = 7.0     # cold cell's credit
        fx.ll_change[1, w0, b0] = 9.0     # hot cell's credit
        fx.prop[1][0, w0, b0] = 3
        fx.prop[1][1, w0, b0] = 5
        n = fx.sweep(parity=0)
        self.assertGreaterEqual(n, 1)
        self.assertEqual(fx.ll_change[0, w0, b0], 9.0)
        self.assertEqual(fx.ll_change[1, w0, b0], 7.0)
        self.assertEqual(fx.prop[1][0, w0, b0], 5)
        self.assertEqual(fx.prop[1][1, w0, b0], 3)

    def test_cell_ll_slot_state_follows_the_model(self):
        """`spec` staleness (hazard 3) -- slot->cell label must move."""
        fx = _SweepFixture(ll=[-100.0, -10.0, -100.0, -100.0])
        spec_before = fx.cell_ll["spec"].copy()
        ll0_before = fx.cell_ll["ll0"].copy()
        n = fx.sweep(parity=0)
        self.assertGreaterEqual(n, 1)
        self.assertEqual(fx.cell_ll["spec"][0], spec_before[1])
        self.assertEqual(fx.cell_ll["spec"][1], spec_before[0])
        self.assertEqual(fx.cell_ll["ll0"][0], ll0_before[1])
        self.assertEqual(fx.cell_ll["ll0"][1], ll0_before[0])

    def test_beta_matches_the_new_temperature(self):
        """Hazard 2: a stale beta would score later repeats wrongly."""
        fx = _SweepFixture(ll=[-100.0, -10.0, -100.0, -100.0])
        fx.sweep(parity=0)
        np.testing.assert_allclose(
            fx.beta, fx.band_temps[fx.b, fx.t],
            err_msg="beta must be re-derived from the post-swap temperature",
        )

    def test_parity_keeps_each_row_in_at_most_one_pair(self):
        """Adjacent pairs overlap; parity is what makes the sweep disjoint."""
        mv = _make_move()
        t, w, b = _rows()
        hot, cold = mv._vertical_pairs(t, w, b)
        for parity in (0, 1):
            sel = (t[cold] % 2) == parity
            rows = np.concatenate([hot[sel], cold[sel]])
            self.assertEqual(len(rows), len(set(rows.tolist())),
                             f"parity {parity} reused a row")

    def test_ladder_is_never_modified(self):
        """_adapt_band_temps stays exclusive to run_tempering."""
        fx = _SweepFixture(ll=[-100.0, -10.0, -100.0, -100.0])
        before = fx.band_temps.copy()
        fx.sweep(parity=0)
        np.testing.assert_array_equal(fx.band_temps, before)


class CellOrderTest(unittest.TestCase):
    """``BandScheduler`` cell ordering, and why vertical swaps need it.

    A vertical pair requires ``(t, w, b)`` and ``(t-1, w, b)`` to be
    resident in the buffer at the SAME time. Under the historical
    count-ordering that is coincidence; ``cell_order="band"`` makes a
    sub-band's whole column contiguous so partners land together.
    """

    NT, NW, NB, SLOTS = 8, 6, 40, 200

    def _cells(self, seed=2, lam=4.0):
        """Synthetic per-cell source counts -> a flat special-index array."""
        rng = np.random.default_rng(seed)
        cnt = rng.poisson(lam, size=(self.NT, self.NW, self.NB))
        t, w, b = np.meshgrid(
            np.arange(self.NT), np.arange(self.NW), np.arange(self.NB),
            indexing="ij",
        )
        spec = pack_special_index(t.ravel(), w.ravel(), b.ravel(), self.NW)
        return np.repeat(spec, cnt.ravel())

    def _resident(self, order):
        from lisatools.globalfit.moves.gbbands import BandScheduler

        sch = BandScheduler(self._cells(), self.SLOTS, xp=np,
                            cell_order=order, nwalkers=self.NW)
        return sch, set(sch.slot_specials.tolist())

    def _pair_fraction(self, resident):
        step = self.NW * 1000000          # one temperature rung
        n = sum(1 for s in resident if (s - step) in resident)
        return n / max(len(resident), 1)

    def test_default_is_count_and_unchanged(self):
        from lisatools.globalfit.moves.gbbands import BandScheduler

        cells = self._cells()
        a = BandScheduler(cells, self.SLOTS, xp=np)
        b = BandScheduler(cells, self.SLOTS, xp=np, cell_order="count")
        np.testing.assert_array_equal(a.cell_specials, b.cell_specials)
        np.testing.assert_array_equal(a.cell_counts, b.cell_counts)
        self.assertEqual(a.cell_order, "count")

    def test_count_order_is_ascending(self):
        sch, _ = self._resident("count")
        self.assertTrue(np.all(np.diff(sch.cell_counts) >= 0))

    def test_band_order_groups_each_band_contiguously(self):
        sch, _ = self._resident("band")
        bands = sch.cell_specials % 1000000
        # a band's entries must form ONE contiguous run
        self.assertTrue(np.all(np.diff(bands) >= 0), "bands not sorted")
        self.assertEqual(len(np.unique(bands)), self.NB)

    def test_band_order_puts_temperature_partners_adjacent(self):
        """Temperature LAST is what survives a buffer smaller than a column.

        A vertical pair is (t, w, b) / (t-1, w, b). Ordering by
        (band, walker, temp) makes those neighbours in the slot sequence, so
        they stay co-resident even when slots < ntemps*nwalkers.
        """
        sch, _ = self._resident("band")
        spec = sch.cell_specials
        tw = spec // 1000000
        walker, temp = tw % self.NW, tw // self.NW
        band = spec % 1000000
        # Within a (band, walker) group temperatures must run in ascending
        # order with no other group interleaved. They need not be
        # CONSECUTIVE: an empty cell has no sources, so it never enters the
        # scheduler and its rung is simply absent.
        step = np.diff(temp)
        same_group = (np.diff(band) == 0) & (np.diff(walker) == 0)
        self.assertTrue(
            np.all(step[same_group] > 0),
            "temperatures within a (band, walker) group must ascend",
        )
        # and every present partner pair IS adjacent in the slot sequence
        adjacent_pairs = int(np.sum(same_group & (step == 1)))
        self.assertGreater(
            adjacent_pairs, 0,
            "no vertical partner pair ended up adjacent -- the ordering is "
            "not delivering the property it exists for",
        )

    def test_band_order_needs_nwalkers(self):
        from lisatools.globalfit.moves.gbbands import BandScheduler

        with self.assertRaises(ValueError):
            BandScheduler(self._cells(), self.SLOTS, xp=np, cell_order="band")

    def test_band_order_preserves_the_cell_set(self):
        """Ordering only -- no cell may be added, dropped or duplicated."""
        sa, _ = self._resident("count")
        sb, _ = self._resident("band")
        np.testing.assert_array_equal(
            np.sort(sa.cell_specials), np.sort(sb.cell_specials))
        self.assertEqual(sa.n_cells, sb.n_cells)
        # counts must travel with their own cell
        ma = dict(zip(sa.cell_specials.tolist(), sa.cell_counts.tolist()))
        mb = dict(zip(sb.cell_specials.tolist(), sb.cell_counts.tolist()))
        self.assertEqual(ma, mb)

    def test_band_order_raises_vertical_partner_availability(self):
        """The measurement that motivates the knob."""
        _, res_count = self._resident("count")
        _, res_band = self._resident("band")
        f_count = self._pair_fraction(res_count)
        f_band = self._pair_fraction(res_band)
        self.assertGreater(
            f_band, 3.0 * f_count,
            f"band ordering must materially raise vertical partner "
            f"availability (count={f_count:.3f}, band={f_band:.3f})",
        )

    def test_bad_order_rejected(self):
        from lisatools.globalfit.moves.gbbands import BandScheduler

        with self.assertRaises(ValueError):
            BandScheduler(self._cells(), self.SLOTS, xp=np,
                          cell_order="sideways", nwalkers=self.NW)


class VerticalWiringTest(unittest.TestCase):
    """The sweep as wired INTO ``_run_in_model_repeats``.

    The unit tests above call ``_vertical_swap_sweep`` directly; this
    exercises the wiring around it -- the per-repeat call, the ``_half_pre``
    rebuild after an accepted swap, and the block-boundary barrier.
    """

    @staticmethod
    def _harness():
        """The in-model fakes, importable as ``tests.x`` or bare ``x``."""
        try:
            from tests import test_inmodel_repeats as h
        except ImportError:  # discovered from inside tests/
            import test_inmodel_repeats as h
        return h

    def _problem(self, n_rep, vertical, seed=11):
        h = self._harness()
        _FakeBuffer, _make_move = h._FakeBuffer, h._make_move

        n_src = 8
        # ONE (walker, band) column, one picked row per temperature: the
        # only shape in which vertical partners exist at all.
        t = np.arange(NTEMPS)
        w = np.zeros(NTEMPS, dtype=int)
        b = np.ones(NTEMPS, dtype=int)
        ids = np.arange(NTEMPS)
        rng = np.random.RandomState(seed)
        coords = np.zeros((n_src, 4))
        coords[:, 0] = rng.uniform(-0.5, 0.5, n_src)
        coords[:, 1] = rng.uniform(2.95, 3.05, n_src)
        coords[:, 2] = rng.uniform(-1, 1, n_src)
        coords[:, 3] = rng.uniform(-1, 1, n_src)

        picked = {
            "ids": ids, "specials": pack_special_index(t, w, b, NWALKERS),
            "slot_index": np.arange(NTEMPS, dtype=np.int32),
            "temp_inds": t.copy(), "walker_inds": w.copy(),
            "band_inds": b.copy(), "N_vals": np.full(NTEMPS, 64),
        }
        sorter = _FakeSorter(t, w, b, NWALKERS)
        sorter.inds = np.ones(n_src, dtype=bool)
        sorter.coords = coords.copy()
        sorter.leaf_inds = np.arange(n_src)

        mv = _make_move(n_rep)
        mv.ntemps, mv.nwalkers, mv.num_bands = NTEMPS, NWALKERS, NBANDS
        mv.temper_vertical = vertical
        mv._temper_rng = np.random.default_rng(5)
        mv.sequential_parity_repeats = False

        band_temps = _ladder()
        ll_change = np.zeros((NTEMPS, NWALKERS, NBANDS))
        prop = np.zeros((2, NTEMPS, NWALKERS, NBANDS), dtype=int)
        acc = np.zeros_like(prop)
        cell_ll = {
            "spec": sorter.special_band_inds.copy(),
            "ll0": np.zeros(NTEMPS), "led0": np.zeros(NTEMPS),
            "rep0": np.zeros(NTEMPS, dtype=int),
        }
        np.random.seed(seed)
        mv._run_in_model_repeats(
            None, sorter, _FakeBuffer(NTEMPS), band_temps, picked,
            ll_change, prop, acc, num_repeats=n_rep, cell_ll_state=cell_ll,
        )
        return mv, sorter, band_temps, picked

    def test_off_by_default_leaves_temperatures_alone(self):
        _, sorter, _, picked = self._problem(6, vertical=False)
        np.testing.assert_array_equal(
            sorter.temp_inds, picked["temp_inds"],
            err_msg="vertical swaps must be OFF unless asked for",
        )

    def test_on_swaps_and_leaves_state_consistent(self):
        mv, sorter, band_temps, picked = self._problem(6, vertical=True)
        # the barrier inside _run_in_model_repeats would have raised on an
        # inconsistent relabel; assert the invariant here too
        self.assertTrue(bool(sorter.special_index_check))
        # rungs are a permutation of the originals (a swap conserves them)
        np.testing.assert_array_equal(
            np.sort(sorter.temp_inds), np.sort(picked["temp_inds"]),
        )
        self.assertFalse(sorter.touched_template_buffer)

    def test_ladder_untouched_by_the_in_model_loop(self):
        _, _, band_temps, _ = self._problem(6, vertical=True)
        np.testing.assert_array_equal(band_temps, _ladder())


class _RealMethodSorter:
    """Duck-typed state carrying ONLY what the real BandSorter label
    methods touch, so the real (unbound) exchange methods can run against
    it on CPU: xp, special_band_inds, temp_inds, walker_inds, band_inds."""

    xp = np

    def __init__(self, t, w, b, nwalkers):
        self.temp_inds = np.asarray(t).copy()
        self.walker_inds = np.asarray(w).copy()
        self.band_inds = np.asarray(b).copy()
        self.nwalkers = nwalkers
        self.special_band_inds = pack_special_index(
            self.temp_inds, self.walker_inds, self.band_inds, nwalkers)


class BatchExchangeEquivalenceTest(unittest.TestCase):
    """exchange_cell_labels_batch == K sequential pairwise calls.

    The orchestration audit (2026-08-27): the vertical sweep called
    exchange_cell_labels once PER ACCEPTED SWAP -- 2 full-table isin + 2
    int() syncs + 2 assert syncs each, ~51 ms/step of the 70 ms repeat
    cost. The batch primitive does ONE membership pass for all K disjoint
    pairs. Equivalence requires the 2K cells to be pairwise disjoint,
    which the sweep's parity selection guarantees.
    """

    def _states(self):
        # 3 pairs across distinct (temp, walker) cells of band 1, walkers
        # 0/1, temps 0..3; several sources per cell + bystander rows.
        t = np.array([0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2])
        w = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        b = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2])
        return (_RealMethodSorter(t, w, b, NWALKERS),
                _RealMethodSorter(t, w, b, NWALKERS))

    def test_batch_matches_sequential(self):
        from lisatools.globalfit.moves.gbbands import BandSorter

        s_seq, s_bat = self._states()
        # pairs: (t=0,w=0,b=1)<->(t=1,w=0,b=1), (t=2,w=0,b=1)<->(t=3,w=0,b=1),
        #        (t=0,w=1,b=2)<->(t=2,w=1,b=2)  -- disjoint cells
        t_h = np.array([1, 3, 2])
        t_c = np.array([0, 2, 0])
        w_p = np.array([0, 0, 1])
        b_p = np.array([1, 1, 2])
        sp_h = pack_special_index(t_h, w_p, b_p, NWALKERS)
        sp_c = pack_special_index(t_c, w_p, b_p, NWALKERS)

        for k in range(3):
            BandSorter.exchange_cell_labels(
                s_seq, sp_h[k:k + 1], int(t_h[k]), w_p[k:k + 1],
                sp_c[k:k + 1], int(t_c[k]), w_p[k:k + 1],
                bands=b_p[k:k + 1])
        BandSorter.exchange_cell_labels_batch(
            s_bat, sp_h, t_h, w_p, sp_c, t_c, w_p, bands=b_p)

        np.testing.assert_array_equal(
            s_seq.special_band_inds, s_bat.special_band_inds)
        np.testing.assert_array_equal(s_seq.temp_inds, s_bat.temp_inds)
        np.testing.assert_array_equal(s_seq.walker_inds, s_bat.walker_inds)
        np.testing.assert_array_equal(s_seq.band_inds, s_bat.band_inds)
        # sanity: something actually moved, and bystanders did not
        self.assertFalse(np.array_equal(
            s_bat.temp_inds, self._states()[0].temp_inds))
        self.assertEqual(int(s_bat.temp_inds[9]), 1)  # untouched bystander

    def test_batch_empty_is_noop(self):
        from lisatools.globalfit.moves.gbbands import BandSorter

        s, ref = self._states()
        e = np.array([], dtype=np.int64)
        BandSorter.exchange_cell_labels_batch(
            s, e, e, e, e, e, e, bands=None)
        np.testing.assert_array_equal(s.temp_inds, ref.temp_inds)
        np.testing.assert_array_equal(
            s.special_band_inds, ref.special_band_inds)


class VerticalCensusDeSyncTest(unittest.TestCase):
    """Rung counters accumulate DEVICE-side per sweep and flush ONCE at the
    block-end log (orchestration audit 2026-08-27 candidate 6): the two
    per-sweep host bincount pulls were 2 forced syncs x one sweep per
    repeat step, spent purely on a log line."""

    def _swept_census(self):
        fx = _SweepFixture(ll=[-100.0, -10.0, -100.0, -100.0])
        cn = fx.mv._vertical_census_new(NTEMPS)
        n_acc = fx.mv._vertical_swap_sweep(
            fx.sorter, fx.band_temps, fx.t, fx.w, fx.b,
            fx.slots, fx.beta, fx.ll_ref, fx.ll_change,
            fx.prop, fx.acc, fx.cell_ll, 0, census=cn,
        )
        return fx, cn, n_acc

    def test_host_rung_arrays_untouched_per_sweep(self):
        _, cn, _ = self._swept_census()
        self.assertGreater(cn["proposed"], 0)
        self.assertEqual(int(cn["prop_by_rung"].sum()), 0)
        self.assertIsNotNone(cn.get("prop_by_rung_dev"))
        self.assertGreater(int(np.asarray(cn["prop_by_rung_dev"]).sum()), 0)

    def test_flush_merges_exactly_once_and_matches_totals(self):
        fx, cn, n_acc = self._swept_census()
        fx.mv._vertical_census_flush(cn)
        self.assertEqual(int(cn["prop_by_rung"].sum()), cn["proposed"])
        self.assertEqual(int(cn["acc_by_rung"].sum()), cn["accepted"])
        self.assertEqual(cn["accepted"], n_acc)
        self.assertIsNone(cn.get("prop_by_rung_dev"))
        self.assertIsNone(cn.get("acc_by_rung_dev"))
        # idempotent: flushing again changes nothing
        before = cn["prop_by_rung"].copy()
        fx.mv._vertical_census_flush(cn)
        np.testing.assert_array_equal(cn["prop_by_rung"], before)


if __name__ == "__main__":
    unittest.main()
