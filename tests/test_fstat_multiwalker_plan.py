"""Multi-walker F-stat fit: reference selection, slot plan, knob, cache dirs.

Plan 7 of the gpu-count-routing work. The epoch fit used to be pinned to ONE
reference walker's residual -- a PROXY for "the residual with the most left
to find", and a lossy one. These are the pieces that let W walkers be fitted
and their peaks unioned; the sweep itself is unchanged and is not exercised
here.

The load-bearing property throughout is **W = 1 bit-identity**: the plural
selector must reproduce the scalar one including tie-breaks, and the per-slot
cache directory must be the epoch directory itself, so every existing cache
key and golden stays valid.
"""

import os
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import (
    GBSpecialBase,
    GBSpecialRJFStatGridMove,
    fstat_fit_plan,
    fstat_plan_is_collective,
)


def _move(search=True, nwalkers=4, root="/tmp/fstat_root"):
    # The F-stat GRID move: ``_epoch_dir`` / ``_fstat_root`` / the fit
    # decision all live on this subclass, so the multi-walker pieces do too.
    m = GBSpecialRJFStatGridMove.__new__(GBSpecialRJFStatGridMove)
    m._fstat_search_residual = bool(search)
    m.nwalkers = int(nwalkers)
    m.name = "rj_fstat_search" if search else "rj_fstat_pe"
    m._fstat_root_value = root
    return m


class RefSelectorTest(unittest.TestCase):
    """``_fstat_fit_refs_from`` is the plural of ``_fstat_fit_ref_from``."""

    LLS = np.array([-10.0, -30.0, -20.0, -40.0])

    def test_search_takes_the_lowest_lnl_first(self):
        # min-lnL = the residual still holding the most unfound signal
        m = _move(search=True)
        self.assertEqual(m._fstat_fit_refs_from(self.LLS, 4), [3, 1, 2, 0])

    def test_pe_takes_the_LOWEST_lnl_first_too(self):
        """2026-09-26: PE joined search on min-lnL. The old PE rule took the
        HIGHEST first, which fitted the grid to the emptiest residual in the
        ensemble. Both modes now agree, and fstat_search_residual selects
        only whether the GB-free window opens."""
        m = _move(search=False)
        self.assertEqual(m._fstat_fit_refs_from(self.LLS, 4),
                         _move(search=True)._fstat_fit_refs_from(self.LLS, 4))

    def test_n_equals_one_reproduces_the_scalar_selector(self):
        for search in (True, False):
            with self.subTest(search=search):
                m = _move(search=search)
                self.assertEqual(m._fstat_fit_refs_from(self.LLS, 1)[0],
                                 m._fstat_fit_ref_from(self.LLS))

    def test_ties_break_the_same_way_as_argmin_argmax(self):
        # THE hinge for W=1 bit-identity: numpy's argmin/argmax take the
        # LOWEST INDEX among equal values, and a stable sort must agree --
        # including on the PE side, where a naive reverse would flip it.
        for lls in (np.zeros(5),
                    np.array([1.0, 1.0, 0.0, 0.0, 1.0]),
                    np.array([-3.0, -3.0, -3.0])):
            for search in (True, False):
                with self.subTest(lls=list(lls), search=search):
                    m = _move(search=search)
                    self.assertEqual(m._fstat_fit_refs_from(lls, 1)[0],
                                     m._fstat_fit_ref_from(lls))

    def test_n_is_clamped_to_the_walker_count(self):
        m = _move(search=True)
        self.assertEqual(len(m._fstat_fit_refs_from(self.LLS, 99)), 4)
        self.assertEqual(len(m._fstat_fit_refs_from(self.LLS, 0)), 1)

    def test_the_refs_are_distinct(self):
        m = _move(search=True)
        refs = m._fstat_fit_refs_from(np.zeros(6), 6)
        self.assertEqual(sorted(refs), list(range(6)))


class FitWalkersKnobTest(unittest.TestCase):
    """Default 1 in BOTH modes (user ruling 2026-09-23).

    The epoch keeps being fitted to the single reference walker -- the
    MIN-lnL cold chain in search, the max in PE -- so the multi-walker union
    is entirely opt-in and the default path is bit-identical to today. The
    cost is why: W walkers is exactly W times the evaluations (no work is
    shared -- each walker's own inverse-PSD row enters ``M_upper``), so W=4
    wants 16 GPUs and would take the 4-GPU epoch ~1100 s -> ~3534 s.
    """

    def test_search_defaults_to_the_single_reference_walker(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_FSTAT_FIT_WALKERS", None)
            self.assertEqual(_move(search=True, nwalkers=4).fstat_fit_walkers, 1)

    def test_pe_defaults_to_one_too(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_FSTAT_FIT_WALKERS", None)
            self.assertEqual(_move(search=False, nwalkers=4).fstat_fit_walkers, 1)

    def test_the_default_is_independent_of_the_walker_count(self):
        # a 24-walker PE run must not silently start fitting 24 grids
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_FSTAT_FIT_WALKERS", None)
            for nw in (1, 4, 24, 32):
                for search in (True, False):
                    with self.subTest(nwalkers=nw, search=search):
                        self.assertEqual(
                            _move(search=search, nwalkers=nw).fstat_fit_walkers, 1)

    def test_the_default_selects_the_min_lnl_walker_in_search(self):
        # the single reference stays the MIN-lnL cold chain: the residual
        # still holding the most unfound signal
        m = _move(search=True, nwalkers=4)
        lls = np.array([-10.0, -30.0, -20.0, -40.0])
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_FSTAT_FIT_WALKERS", None)
            refs = m._fstat_fit_refs_from(lls, m.fstat_fit_walkers)
        self.assertEqual(refs, [3])
        self.assertEqual(refs[0], int(np.argmin(lls)))

    def test_an_explicit_value_overrides_both_defaults(self):
        with mock.patch.dict(os.environ, {"GB_FSTAT_FIT_WALKERS": "2"}):
            self.assertEqual(_move(search=True).fstat_fit_walkers, 2)
            self.assertEqual(_move(search=False).fstat_fit_walkers, 2)

    def test_one_is_the_rollback(self):
        with mock.patch.dict(os.environ, {"GB_FSTAT_FIT_WALKERS": "1"}):
            self.assertEqual(_move(search=True).fstat_fit_walkers, 1)

    def test_garbage_falls_back_to_the_default_without_raising(self):
        with mock.patch.dict(os.environ, {"GB_FSTAT_FIT_WALKERS": "lots"}):
            self.assertEqual(_move(search=False).fstat_fit_walkers, 1)


class _StubLayout:
    """Just the pieces ``fstat_fit_plan`` reads."""

    def __init__(self, compute_ranks, block=1, head=0):
        self.compute_ranks = tuple(compute_ranks)
        self.block = int(block)
        self.head_rank = int(head)

    def owner_of(self, w):
        return int(self.compute_ranks[int(w) // self.block]), int(w) % self.block


class FitPlanTest(unittest.TestCase):
    def test_one_slot_per_reference_covering_every_rank(self):
        lay = _StubLayout((0, 1, 2, 3), block=1)
        plan = fstat_fit_plan(lay, [3, 1])
        self.assertEqual(len(plan), 2)
        self.assertEqual([s.walker for s in plan], [3, 1])
        # the ranks partition, contiguously, with no rank used twice
        used = [r for s in plan for r in s.group_ranks]
        self.assertEqual(sorted(used), [0, 1, 2, 3])
        self.assertEqual(plan[0].group_ranks, (0, 1))
        self.assertEqual(plan[1].group_ranks, (2, 3))

    def test_w_equals_one_gives_every_rank_to_the_single_fit(self):
        lay = _StubLayout((0, 1, 2, 3), block=1)
        plan = fstat_fit_plan(lay, [2])
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].group_ranks, (0, 1, 2, 3))
        self.assertTrue(plan[0].needs_bcast)

    def test_a_singleton_group_needs_no_broadcast(self):
        lay = _StubLayout((0, 1, 2, 3), block=1)
        plan = fstat_fit_plan(lay, [0, 1, 2, 3])
        self.assertTrue(all(len(s.group_ranks) == 1 for s in plan))
        self.assertTrue(all(not s.needs_bcast for s in plan))
        self.assertFalse(fstat_plan_is_collective(plan))

    def test_the_collective_decision_is_per_command_not_per_rank(self):
        # 3 refs over 4 ranks -> groups of 2, 1, 1. Mixing a Bcast-ing group
        # with a returning one DEADLOCKS, so the whole command must be
        # collective if ANY slot is.
        lay = _StubLayout((0, 1, 2, 3), block=1)
        plan = fstat_fit_plan(lay, [0, 1, 2])
        sizes = [len(s.group_ranks) for s in plan]
        self.assertEqual(sizes, [2, 1, 1])
        self.assertTrue(any(s.needs_bcast for s in plan))
        self.assertFalse(all(s.needs_bcast for s in plan))
        self.assertTrue(fstat_plan_is_collective(plan))

    def test_owner_and_local_index_come_from_the_layout(self):
        # 2 blocks of 2 walkers: walker 3 lives on rank 1 at local row 1
        lay = _StubLayout((0, 1), block=2)
        plan = fstat_fit_plan(lay, [3])
        self.assertEqual((plan[0].owner_rank, plan[0].local_index), (1, 1))

    def test_four_walkers_two_replicas_fit_on_all_eight_ranks(self):
        """``NGPUS=8 GF_GPU_ROUTING=1`` (4 blocks x R = 2, the real layout):
        the epoch fit's node ranges spread over ALL 8 compute ranks -- twice
        the 4-GPU run's -- and the reference row comes from its block's LEAD
        (``owner_of``), local row 0."""
        from lisatools.globalfit.communication.fakecomm import FakeWorld
        from lisatools.globalfit.communication.ranks import build_layout

        lay = FakeWorld(9, nodes=[0, 1, 2, 3, 0, 1, 2, 3, 0]).run(
            lambda r, c: build_layout(c, 4, [0, 1], legacy=False,
                                      gpu_routing=True))[0]
        plan = fstat_fit_plan(lay, [2])
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].group_ranks, tuple(range(8)))
        self.assertEqual((plan[0].owner_rank, plan[0].local_index), (4, 0))
        self.assertTrue(fstat_plan_is_collective(plan))

    def test_no_layout_is_the_single_process_plan(self):
        plan = fstat_fit_plan(None, [2, 0])
        self.assertEqual([s.walker for s in plan], [2, 0])
        self.assertTrue(all(s.group_ranks == (0,) for s in plan))
        self.assertFalse(fstat_plan_is_collective(plan))

    def test_it_reuses_split_box_range(self):
        # not a second contiguous partition that merely agrees today
        from lisatools.sampling.fstat_gridfit import split_box_range
        lay = _StubLayout(tuple(range(7)), block=1)
        plan = fstat_fit_plan(lay, [0, 1, 2])
        expected = [tuple(range(int(a), int(b)))
                    for a, b in split_box_range(0, 7, 3)]
        self.assertEqual([s.group_ranks for s in plan], expected)


class SlotDirTest(unittest.TestCase):
    def _m(self):
        m = _move(search=True)
        m._epoch_dir = lambda k: f"/root/epoch_{k:04d}"
        return m

    def test_one_slot_is_the_epoch_dir_itself(self):
        # the filesystem half of W=1 bit-identity: no w## level appears
        m = self._m()
        self.assertEqual(m._fstat_slot_dir(4, 3, 1), "/root/epoch_0004")

    def test_several_slots_are_named_by_global_walker(self):
        m = self._m()
        self.assertEqual(m._fstat_slot_dir(4, 3, 2), "/root/epoch_0004/w03")
        self.assertEqual(m._fstat_slot_dir(4, 11, 2), "/root/epoch_0004/w11")

    def test_the_name_does_not_depend_on_slot_position(self):
        # a mid-epoch restart that reshuffles the lnL ordering among the SAME
        # walkers must hit all W caches, not none
        m = self._m()
        first = [m._fstat_slot_dir(7, w, 3) for w in (2, 0, 1)]
        after_reshuffle = [m._fstat_slot_dir(7, w, 3) for w in (0, 1, 2)]
        self.assertEqual(sorted(first), sorted(after_reshuffle))


if __name__ == "__main__":
    unittest.main()


class UnionStackedNpzTest(unittest.TestCase):
    """Union several slots' stage-B grids into one birth cache."""

    @staticmethod
    def _slot(path, *, n_boxes=3, n_groups=1, f0_base=1.0, basis="mc",
              alpha=(0.0, 1.0), seed=0):
        import numpy as _np
        rng = _np.random.default_rng(seed)
        per = n_boxes // n_groups
        sizes = [per] * n_groups
        sizes[-1] += n_boxes - per * n_groups
        d = dict(
            f0_los=_np.arange(n_boxes, dtype=float) + f0_base,
            f0_dxs=_np.full(n_boxes, 0.5),
            alpha_ax=_np.asarray(alpha, dtype=float),
            sin_delta_ax=_np.asarray([-1.0, 1.0]),
            grid_basis=_np.asarray(basis),
            grid_c_t=_np.asarray(2.0),
            peak_f0_mHz=_np.arange(n_boxes, dtype=float) + f0_base,
            peak_F=rng.random(n_boxes) * 100.0,
            band_idx=_np.zeros(n_boxes, dtype=int),
            band_f0_lo=_np.zeros(n_boxes),
            band_f0_hi=_np.ones(n_boxes),
            band_edges=_np.asarray([0.0, 1.0, 2.0]),
            group_sizes=_np.asarray(sizes, dtype=int),
        )
        off = 0
        for gi, sz in enumerate(sizes):
            d[f"logp_grids_g{gi}"] = rng.random((sz, 2, 2, 2))
            d[f"mc_ax_g{gi}"] = _np.asarray([1.0, 2.0])
            off += sz
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **d)
        return d

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory(prefix="fstat_union_")
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_boxes_and_groups_are_concatenated(self):
        from lisatools.sampling.fstat_gridfit import union_stacked_npz
        a = os.path.join(self.root, "w00", "g.npz")
        b = os.path.join(self.root, "w01", "g.npz")
        self._slot(a, n_boxes=3, n_groups=1, f0_base=1.0, seed=1)
        self._slot(b, n_boxes=4, n_groups=2, f0_base=10.0, seed=2)
        out = os.path.join(self.root, "union.npz")
        total, ngroups = union_stacked_npz([a, b], out)
        self.assertEqual((total, ngroups), (7, 3))
        z = np.load(out)
        self.assertEqual(list(z["group_sizes"]), [3, 2, 2])
        self.assertEqual(len(z["f0_los"]), 7)
        # every group survived under a fresh sequential number
        for gi in range(3):
            self.assertIn(f"logp_grids_g{gi}", z.files)
            self.assertIn(f"mc_ax_g{gi}", z.files)
        self.assertNotIn("logp_grids_g3", z.files)

    def test_grids_are_carried_through_unchanged(self):
        from lisatools.sampling.fstat_gridfit import union_stacked_npz
        a = os.path.join(self.root, "w00", "g.npz")
        b = os.path.join(self.root, "w01", "g.npz")
        da = self._slot(a, n_boxes=2, n_groups=1, seed=3)
        db = self._slot(b, n_boxes=2, n_groups=1, f0_base=9.0, seed=4)
        out = os.path.join(self.root, "u.npz")
        union_stacked_npz([a, b], out)
        z = np.load(out)
        # the union must not recompute anything
        np.testing.assert_array_equal(z["logp_grids_g0"], da["logp_grids_g0"])
        np.testing.assert_array_equal(z["logp_grids_g1"], db["logp_grids_g0"])

    def test_box_provenance_is_recorded(self):
        from lisatools.sampling.fstat_gridfit import union_stacked_npz
        a = os.path.join(self.root, "w00", "g.npz")
        b = os.path.join(self.root, "w01", "g.npz")
        self._slot(a, n_boxes=2, seed=5)
        self._slot(b, n_boxes=3, f0_base=9.0, seed=6)
        out = os.path.join(self.root, "u.npz")
        union_stacked_npz([a, b], out)
        self.assertEqual(list(np.load(out)["box_slot"]), [0, 0, 1, 1, 1])

    def test_one_slot_is_a_faithful_passthrough(self):
        from lisatools.sampling.fstat_gridfit import union_stacked_npz
        a = os.path.join(self.root, "w00", "g.npz")
        d = self._slot(a, n_boxes=3, n_groups=2, seed=7)
        out = os.path.join(self.root, "u.npz")
        total, ngroups = union_stacked_npz([a], out)
        self.assertEqual((total, ngroups), (3, 2))
        z = np.load(out)
        np.testing.assert_array_equal(z["f0_los"], d["f0_los"])
        np.testing.assert_array_equal(z["group_sizes"], d["group_sizes"])

    def test_a_disagreeing_sky_axis_is_refused(self):
        from lisatools.sampling.fstat_gridfit import union_stacked_npz
        a = os.path.join(self.root, "w00", "g.npz")
        b = os.path.join(self.root, "w01", "g.npz")
        self._slot(a, seed=8)
        self._slot(b, alpha=(0.0, 2.0), seed=9)     # different alpha axis
        with self.assertRaisesRegex(ValueError, "alpha_ax differs"):
            union_stacked_npz([a, b], os.path.join(self.root, "u.npz"))

    def test_a_disagreeing_basis_is_refused(self):
        from lisatools.sampling.fstat_gridfit import union_stacked_npz
        a = os.path.join(self.root, "w00", "g.npz")
        b = os.path.join(self.root, "w01", "g.npz")
        self._slot(a, basis="mc", seed=10)
        self._slot(b, basis="fdot", seed=11)
        with self.assertRaisesRegex(ValueError, "basis"):
            union_stacked_npz([a, b], os.path.join(self.root, "u.npz"))

    def test_no_paths_raises(self):
        from lisatools.sampling.fstat_gridfit import union_stacked_npz
        with self.assertRaises(ValueError):
            union_stacked_npz([], os.path.join(self.root, "u.npz"))


class UnionCellKeysTest(unittest.TestCase):
    """The cell census must count UNIQUE peaks, by an exact key."""

    def test_same_node_same_band_shares_a_key(self):
        from lisatools.sampling.fstat_gridfit import union_cell_keys
        band = np.array([0, 0, 0, 1])
        f0 = np.array([1.5, 1.5, 2.5, 1.5])
        k = union_cell_keys(band, f0)
        self.assertEqual(k[0], k[1])          # same band, same node
        self.assertNotEqual(k[0], k[2])       # same band, different node
        self.assertNotEqual(k[0], k[3])       # same node, different band

    def test_w_copies_of_one_peak_collapse_to_one_cell(self):
        # the whole point: 4 walkers' copies of a contested peak must not
        # read as 4 peaks and get down-weighted
        from lisatools.sampling.fstat_gridfit import union_cell_keys
        band = np.zeros(4, dtype=int)
        f0 = np.full(4, 3.25)
        self.assertEqual(len(set(union_cell_keys(band, f0).tolist())), 1)

    def test_it_is_exact_not_a_tolerance(self):
        # a neighbouring node is a DIFFERENT cell, however close
        from lisatools.sampling.fstat_gridfit import union_cell_keys
        band = np.zeros(2, dtype=int)
        f0 = np.array([1.0, np.nextafter(1.0, 2.0)])
        k = union_cell_keys(band, f0)
        self.assertNotEqual(k[0], k[1])

    def test_shape_mismatch_raises(self):
        from lisatools.sampling.fstat_gridfit import union_cell_keys
        with self.assertRaises(ValueError):
            union_cell_keys(np.zeros(3, int), np.zeros(2))
