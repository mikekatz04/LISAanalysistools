"""Per-walker start index / direction, unit repeats, and the in-model cap headroom.

Three knobbed changes to the search-stage band-unit sweep in
``GBSpecialStretchMove.run_proposal``, all defaulting to today's
behaviour:

1. **Per-walker START index** (``{BRANCH}_BAND_UNIT_START_PER_WALKER``).
   The sweep visits the ``band_units`` residue classes
   (``band_index % units``) IN ORDER starting from one globally drawn
   ``start_unit``. The knob promotes that scalar start to a per-walker
   vector. The classes are still visited IN ORDER -- this is a cyclic
   ROTATION per walker, never a scrambled permutation.

2. **Per-walker DIRECTION** (``{BRANCH}_BAND_UNIT_DIR_PER_WALKER``): the
   per-walker cycle may run ``-1`` instead of ``+1``.

3. **Unit REPEATS** (``{BRANCH}_BAND_UNIT_REPEATS``, default 1,
   search-only): each class is opened/closed ``N`` times consecutively
   before the sweep advances to the next class.

Plus the in-model cap-drift gate now goes through
:meth:`GBSpecialBase._cap_new_entry_veto`, which is where the
already-existing ``GB_CAP_INMODEL_HEADROOM`` (default 2) lives.

⚠ THE DETAILED-BALANCE CONSTRAINT. A random-scan / random-rotation sweep
over blocks preserves stationarity ONLY if the order is drawn UNIFORMLY
and INDEPENDENTLY OF THE CURRENT CHAIN STATE. Choosing a walker's start
or direction by any heuristic (which walker looks stuck, by logL, by band
occupancy) silently converts a DB-safe change into a DB-breaking one.
:class:`UniformStateIndependentDrawTest` is the guard: it pins that the
draw site takes NO state argument, that its body reads nothing but the
RNG, and that starts/directions come out uniform.
"""

import ast
import inspect
import os
import unittest

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import (
    _draw_unit_scan_schedule,
    _format_unit_scan_schedule,
    _ortho_boundary_pairs,
    _resolve_band_unit_dir_per_walker,
    _resolve_band_unit_repeats,
    _resolve_band_unit_start_per_walker,
    _unit_pass_remainder,
    _unit_residue_mask,
    _unit_sweep_passes,
    GBSpecialBase,
)


def _clear(*names):
    for n in names:
        os.environ.pop(n, None)


class _CountingRandom:
    """``model.random`` stand-in that records every RNG method call."""

    def __init__(self, seed=42):
        self._rs = np.random.RandomState(seed)
        self.calls = []

    def randint(self, *args, **kwargs):
        self.calls.append(("randint", args, tuple(sorted(kwargs.items()))))
        return self._rs.randint(*args, **kwargs)

    def permutation(self, *args, **kwargs):
        self.calls.append(("permutation", args, tuple(sorted(kwargs.items()))))
        return self._rs.permutation(*args, **kwargs)

    def random(self, *args, **kwargs):
        self.calls.append(("random", args, tuple(sorted(kwargs.items()))))
        return self._rs.random_sample(*args, **kwargs)


# ---------------------------------------------------------------------------
# Knob resolution
# ---------------------------------------------------------------------------


class ResolveKnobsTest(unittest.TestCase):
    def setUp(self):
        for n in (
            "GB_BAND_UNIT_START_PER_WALKER",
            "GB_BAND_UNIT_DIR_PER_WALKER",
            "GB_BAND_UNIT_REPEATS",
            "VGB_BAND_UNIT_START_PER_WALKER",
            "VGB_BAND_UNIT_DIR_PER_WALKER",
            "VGB_BAND_UNIT_REPEATS",
        ):
            os.environ.pop(n, None)
            self.addCleanup(os.environ.pop, n, None)

    def test_start_and_dir_default_off(self):
        self.assertFalse(_resolve_band_unit_start_per_walker("gb", None))
        self.assertFalse(_resolve_band_unit_dir_per_walker("gb", None))

    def test_env_arms_per_branch(self):
        os.environ["GB_BAND_UNIT_START_PER_WALKER"] = "1"
        self.assertTrue(_resolve_band_unit_start_per_walker("gb", None))
        # branch-prefixed: VGB is untouched by the GB knob
        self.assertFalse(_resolve_band_unit_start_per_walker("vgb", None))
        os.environ["VGB_BAND_UNIT_DIR_PER_WALKER"] = "true"
        self.assertTrue(_resolve_band_unit_dir_per_walker("vgb", None))
        self.assertFalse(_resolve_band_unit_dir_per_walker("gb", None))

    def test_env_wins_over_ctor_for_order_knobs(self):
        os.environ["GB_BAND_UNIT_START_PER_WALKER"] = "0"
        self.assertFalse(_resolve_band_unit_start_per_walker("gb", True))

    def test_repeats_default_one(self):
        self.assertEqual(_resolve_band_unit_repeats("gb", None), 1)

    def test_repeats_env(self):
        os.environ["GB_BAND_UNIT_REPEATS"] = "3"
        self.assertEqual(_resolve_band_unit_repeats("gb", None), 3)

    def test_repeats_explicit_kwarg_wins_over_env(self):
        # the SEARCH-ONLY pin: recipe passes band_unit_repeats=1 to the
        # pe-named moves, and that must beat an exported env.
        os.environ["GB_BAND_UNIT_REPEATS"] = "4"
        self.assertEqual(_resolve_band_unit_repeats("gb", 1), 1)

    def test_repeats_rejects_below_one(self):
        with self.assertRaises(ValueError):
            _resolve_band_unit_repeats("gb", 0)


# ---------------------------------------------------------------------------
# ⚠ Detailed balance: the draw must be uniform and state-independent
# ---------------------------------------------------------------------------


class UniformStateIndependentDrawTest(unittest.TestCase):
    """The one thing that would break detailed balance if it went wrong."""

    ALLOWED_PARAMS = {
        "random_state",
        "nwalkers",
        "units",
        "per_walker_start",
        "per_walker_dir",
    }

    def test_signature_admits_no_chain_state(self):
        params = set(inspect.signature(_draw_unit_scan_schedule).parameters)
        self.assertEqual(
            params,
            self.ALLOWED_PARAMS,
            "the scan-order draw must take ONLY the RNG and the sweep shape; "
            "any state argument (model / state / band_sorter / log_like / "
            "occupancy) would let the order depend on the chain and break "
            "stationarity.",
        )

    def test_body_reads_nothing_but_the_rng(self):
        tree = ast.parse(inspect.getsource(_draw_unit_scan_schedule))
        fn = tree.body[0]
        bound = set(self.ALLOWED_PARAMS)
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            elif isinstance(node, ast.comprehension) and isinstance(
                node.target, ast.Name
            ):
                bound.add(node.target.id)
        free = {
            node.id
            for node in ast.walk(fn)
            if isinstance(node, ast.Name) and node.id not in bound
        }
        self.assertTrue(
            free <= {"np", "int", "bool", "range", "len"},
            f"the scan-order draw reads unexpected globals {sorted(free)}; it "
            "must be a pure function of the RNG and the sweep shape.",
        )

    def test_off_consumes_exactly_the_legacy_single_randint(self):
        rng = _CountingRandom(0)
        starts, dirs = _draw_unit_scan_schedule(
            rng, 24, 9, per_walker_start=False, per_walker_dir=False
        )
        self.assertEqual(
            [c[0] for c in rng.calls],
            ["randint"],
            "with both knobs OFF the draw must consume exactly the legacy "
            "single model.random.randint(units) so the RNG stream -- and "
            "therefore the whole propose -- stays bit-identical.",
        )
        self.assertEqual(rng.calls[0][1], (9,))
        self.assertEqual(len(set(starts.tolist())), 1)
        np.testing.assert_array_equal(dirs, np.ones(24, dtype=int))

    def test_starts_uniform_over_classes(self):
        units, nwalkers, ndraw = 9, 8, 4000
        rng = np.random.RandomState(12345)
        counts = np.zeros(units, dtype=int)
        for _ in range(ndraw):
            starts, _ = _draw_unit_scan_schedule(
                rng, nwalkers, units, per_walker_start=True, per_walker_dir=False
            )
            counts += np.bincount(np.asarray(starts), minlength=units)
        expect = ndraw * nwalkers / units
        chi2 = float(((counts - expect) ** 2 / expect).sum())
        # 8 dof; 26.12 is the 0.999 quantile
        self.assertLess(chi2, 26.12, f"start draw not uniform: {counts}")

    def test_directions_uniform(self):
        rng = np.random.RandomState(999)
        n_pos = n_tot = 0
        for _ in range(2000):
            _, dirs = _draw_unit_scan_schedule(
                rng, 8, 9, per_walker_start=False, per_walker_dir=True
            )
            d = np.asarray(dirs)
            self.assertTrue(set(np.unique(d).tolist()) <= {-1, 1})
            n_pos += int((d == 1).sum())
            n_tot += d.size
        frac = n_pos / n_tot
        self.assertGreater(frac, 0.47)
        self.assertLess(frac, 0.53)

    def test_seed_reproducible(self):
        a = _draw_unit_scan_schedule(
            np.random.RandomState(7), 24, 9, True, True
        )
        b = _draw_unit_scan_schedule(
            np.random.RandomState(7), 24, 9, True, True
        )
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_walkers_actually_differ_when_armed(self):
        rng = np.random.RandomState(3)
        starts, _ = _draw_unit_scan_schedule(rng, 24, 9, True, False)
        self.assertGreater(
            len(set(np.asarray(starts).tolist())),
            1,
            "per-walker starts armed but every walker got the same start",
        )


# ---------------------------------------------------------------------------
# The rotation itself: still IN ORDER, still a partition
# ---------------------------------------------------------------------------


class RotationTest(unittest.TestCase):
    def test_off_reproduces_the_legacy_rotation(self):
        units = 9
        starts = np.full(6, 4, dtype=int)
        dirs = np.ones(6, dtype=int)
        for unit_i in range(units):
            rem = _unit_pass_remainder(starts, dirs, unit_i, units)
            np.testing.assert_array_equal(
                rem, np.full(6, (4 + unit_i) % units)
            )

    def test_classes_visited_in_order(self):
        units = 9
        starts = np.array([2, 7], dtype=int)
        dirs = np.array([1, -1], dtype=int)
        seq = np.array(
            [_unit_pass_remainder(starts, dirs, i, units) for i in range(units)]
        )
        np.testing.assert_array_equal(
            seq[:, 0], [(2 + i) % units for i in range(units)]
        )
        np.testing.assert_array_equal(
            seq[:, 1], [(7 - i) % units for i in range(units)]
        )

    def test_every_walker_visits_every_class_exactly_once(self):
        units = 9
        rng = np.random.RandomState(11)
        starts, dirs = _draw_unit_scan_schedule(rng, 24, units, True, True)
        seq = np.array(
            [_unit_pass_remainder(starts, dirs, i, units) for i in range(units)]
        )
        for w in range(24):
            self.assertEqual(
                sorted(seq[:, w].tolist()),
                list(range(units)),
                "the partition property is what keeps every source opened "
                "exactly once per sweep",
            )


# ---------------------------------------------------------------------------
# The residue mask
# ---------------------------------------------------------------------------


class ResidueMaskTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(5)
        self.units = 9
        self.band_inds = rng.randint(0, 40, size=500)
        self.walker_inds = rng.randint(0, 6, size=500)

    def test_scalar_matches_legacy_expression(self):
        for rem in range(self.units):
            got = _unit_residue_mask(
                self.band_inds, self.walker_inds, self.units, rem
            )
            np.testing.assert_array_equal(
                got, self.band_inds % self.units == rem
            )

    def test_per_walker_mask(self):
        rem = np.array([0, 3, 5, 8, 1, 2])
        got = _unit_residue_mask(
            self.band_inds, self.walker_inds, self.units, rem
        )
        np.testing.assert_array_equal(
            got, self.band_inds % self.units == rem[self.walker_inds]
        )

    def test_degenerate_per_walker_equals_scalar(self):
        """Knob ON with a degenerate schedule == knob OFF, mask for mask."""
        rem_vec = np.full(6, 4)
        np.testing.assert_array_equal(
            _unit_residue_mask(
                self.band_inds, self.walker_inds, self.units, rem_vec
            ),
            _unit_residue_mask(
                self.band_inds, self.walker_inds, self.units, 4
            ),
        )

    def test_sweep_selects_every_source_exactly_once(self):
        rng = np.random.RandomState(21)
        starts, dirs = _draw_unit_scan_schedule(rng, 6, self.units, True, True)
        hits = np.zeros(self.band_inds.size, dtype=int)
        for unit_i in range(self.units):
            rem = _unit_pass_remainder(starts, dirs, unit_i, self.units)
            hits += _unit_residue_mask(
                self.band_inds, self.walker_inds, self.units, rem
            ).astype(int)
        np.testing.assert_array_equal(hits, np.ones_like(hits))


# ---------------------------------------------------------------------------
# Repeats
# ---------------------------------------------------------------------------


class SweepPassesTest(unittest.TestCase):
    def test_repeats_one_is_the_legacy_sweep(self):
        self.assertEqual(
            list(_unit_sweep_passes(9, 1)), [(i, 0) for i in range(9)]
        )

    def test_each_class_repeated_consecutively(self):
        got = list(_unit_sweep_passes(3, 4))
        self.assertEqual(
            [u for u, _ in got],
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            "each residue class must repeat N times BEFORE the sweep "
            "advances to the next class",
        )
        self.assertEqual([r for _, r in got], [0, 1, 2, 3] * 3)

    def test_pass_count_is_linear_in_repeats(self):
        for n in (1, 2, 3, 5):
            self.assertEqual(len(list(_unit_sweep_passes(9, n))), 9 * n)


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


class ScanScheduleLogLineTest(unittest.TestCase):
    def test_line_is_greppable_and_single_line(self):
        starts = np.array([0, 3, 5, 8])
        dirs = np.array([1, -1, 1, -1])
        line = _format_unit_scan_schedule(starts, dirs, units=9, repeats=2)
        self.assertNotIn("\n", line)
        self.assertIn("[GB_UNIT_SCAN]", line)
        self.assertIn("units=9", line)
        self.assertIn("repeats=2", line)
        # the per-walker (start, direction) pairs are recoverable
        self.assertIn("0+", line)
        self.assertIn("3-", line)
        self.assertIn("8-", line)

    def test_large_walker_counts_stay_compact(self):
        rng = np.random.RandomState(1)
        starts = rng.randint(0, 9, size=512)
        dirs = np.where(rng.randint(0, 2, size=512) == 0, -1, 1)
        line = _format_unit_scan_schedule(starts, dirs, units=9, repeats=1)
        self.assertNotIn("\n", line)
        self.assertLess(len(line), 400)
        self.assertIn("digest=", line)

    def test_global_schedule_is_named_too(self):
        line = _format_unit_scan_schedule(
            np.full(24, 4), np.ones(24, dtype=int), units=9, repeats=1
        )
        self.assertIn("[GB_UNIT_SCAN]", line)
        self.assertIn("global", line)


# ---------------------------------------------------------------------------
# The orthogonality-premise sampler takes the per-walker remainder
# ---------------------------------------------------------------------------


class OrthoBoundaryPairsPerWalkerTest(unittest.TestCase):
    def _inputs(self):
        # two walkers, four bands, two sources per band
        band_inds = np.array([0, 0, 1, 1, 2, 2, 3, 3] * 2)
        walker_inds = np.array([0] * 8 + [1] * 8)
        f0 = np.arange(16, dtype=float) * 1e-4
        f0[8:] = np.arange(8, dtype=float) * 1e-4
        eligible = np.ones(16, dtype=bool)
        return f0, walker_inds, band_inds, eligible

    def test_array_remainder_matches_scalar_when_constant(self):
        f0, w, b, e = self._inputs()
        i_s, j_s = _ortho_boundary_pairs(f0, w, b, e, 2, 0)
        i_a, j_a = _ortho_boundary_pairs(f0, w, b, e, 2, np.array([0, 0]))
        np.testing.assert_array_equal(i_s, i_a)
        np.testing.assert_array_equal(j_s, j_a)

    def test_per_walker_remainder_selects_per_walker_classes(self):
        f0, w, b, e = self._inputs()
        i_x, j_x = _ortho_boundary_pairs(f0, w, b, e, 2, np.array([0, 1]))
        rows = np.concatenate([i_x, j_x])
        if rows.size:
            expect = np.array([0, 1])[w[rows]]
            np.testing.assert_array_equal(b[rows] % 2, expect)


# ---------------------------------------------------------------------------
# In-model cap headroom (cap + 2) -- the dead knob gets wired
# ---------------------------------------------------------------------------


class _CapStub:
    """Minimal stand-in exposing what ``_cap_new_entry_veto`` touches."""

    cap_overlap_frac = 0.0

    def __init__(self, nwalkers=2, ntemps=1, num_cap_cells=4):
        self.nwalkers = nwalkers
        self.ntemps = ntemps
        self.num_cap_cells = num_cap_cells

    _cap_flat_index = GBSpecialBase._cap_flat_index
    _cap_new_entry_veto = GBSpecialBase._cap_new_entry_veto


class CapInModelHeadroomTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(os.environ.pop, "GB_CAP_INMODEL_HEADROOM", None)
        os.environ.pop("GB_CAP_INMODEL_HEADROOM", None)

    def _veto(self, counts_new_cell, cap_value):
        stub = _CapStub()
        cells = 4
        counts = np.zeros(stub.ntemps * stub.nwalkers * cells, dtype=int)
        t = np.array([0])
        w = np.array([0])
        cur = (np.array([0]), None, None)
        new = (np.array([1]), None, None)
        flat = stub._cap_flat_index(t, w, new[0])
        counts[flat] = counts_new_cell
        cap = np.full(cells, cap_value, dtype=int)
        return bool(
            stub._cap_new_entry_veto(counts, cap, t, w, cur, new)[0]
        )

    def test_default_headroom_is_two(self):
        # occupancy exactly AT cap: allowed under the default headroom
        self.assertFalse(self._veto(4, 4))
        self.assertFalse(self._veto(5, 4))
        self.assertTrue(self._veto(6, 4))

    def test_headroom_zero_is_the_strict_gate(self):
        os.environ["GB_CAP_INMODEL_HEADROOM"] = "0"
        self.assertTrue(self._veto(4, 4))
        self.assertFalse(self._veto(3, 4))

    def test_in_model_repeats_calls_the_shared_veto(self):
        src = inspect.getsource(GBSpecialBase._run_in_model_repeats)
        self.assertIn(
            "_cap_new_entry_veto(",
            src,
            "the in-model cap drift gate must go through the shared veto "
            "operator (which is where GB_CAP_INMODEL_HEADROOM lives) rather "
            "than carry its own inline copy without the headroom term.",
        )
        self.assertNotIn(
            "_dg_counts[_dg_flat_n] >= _dg_cap[_dg_cell_n]",
            src,
            "the inline no-headroom copy of the veto is still there",
        )

    def test_replace_veto_is_not_gated_on_cap_divisor(self):
        src = inspect.getsource(GBSpecialBase._run_replace_step)
        self.assertNotIn(
            "self.cap_divisor > 1",
            src,
            "the destination-headroom gate must apply at cap_divisor == 1 "
            "too (the production configuration)",
        )

    def test_veto_docstring_no_longer_overclaims(self):
        doc = GBSpecialBase._cap_new_entry_veto.__doc__
        self.assertIn("in-model", doc.lower())

    def test_kernel_cap_mirrors_the_headroom(self):
        from lisatools.globalfit.moves.gbspecialstretch import (
            _cap_with_inmodel_headroom,
        )

        cap = np.array([-1, 0, 3, 7])
        np.testing.assert_array_equal(
            _cap_with_inmodel_headroom(cap),
            np.array([-1, 2, 5, 9]),
        )
        os.environ["GB_CAP_INMODEL_HEADROOM"] = "0"
        np.testing.assert_array_equal(_cap_with_inmodel_headroom(cap), cap)


# ---------------------------------------------------------------------------
# Structural pins: knob-OFF bit-identity, and the search-only repeat pin
# ---------------------------------------------------------------------------


class WiringTest(unittest.TestCase):
    def test_knob_off_keeps_the_legacy_scalar_sweep_call(self):
        """OFF must be bit-identical BY CONSTRUCTION, not by coincidence.

        With both order knobs off the sweep collapses to the literal
        legacy call shape -- a scalar ``remainder`` passed as
        ``units=``/``remainder=`` -- so no new array path can perturb the
        subsets it selects.
        """
        src = inspect.getsource(GBSpecialBase.run_proposal)
        self.assertIn("dict(units=units, remainder=remainder)", src)
        self.assertIn("_unit_sweep_passes(units, _unit_repeats)", src)
        self.assertIn("_draw_unit_scan_schedule(", src)

    def test_partition_guard_accepts_every_legal_schedule(self):
        from lisatools.globalfit.moves.gbspecialstretch import (
            _assert_unit_scan_partition,
        )

        rng = np.random.RandomState(4)
        for _ in range(50):
            starts, dirs = _draw_unit_scan_schedule(rng, 24, 9, True, True)
            _assert_unit_scan_partition(starts, dirs, 9)

    def test_partition_guard_is_loud_on_a_broken_schedule(self):
        from lisatools.globalfit.moves.gbspecialstretch import (
            _assert_unit_scan_partition,
        )

        # direction 0 = a walker stuck on one class all sweep: every other
        # class never opens, so sources in them are never proposed.
        with self.assertRaises(RuntimeError) as ctx:
            _assert_unit_scan_partition(
                np.array([0, 2]), np.array([1, 0]), 9
            )
        self.assertIn("etailed balance", str(ctx.exception))

    def test_run_proposal_verifies_the_partition(self):
        src = inspect.getsource(GBSpecialBase.run_proposal)
        self.assertIn("_assert_unit_scan_partition(", src)

    def test_edge_leak_knob_keeps_the_drift_gate_armed(self):
        """cells == bands does NOT mean cell identity is fixed.

        In-model f0 moves are windowed to the sub-band widened by N/4
        bins per side, so a leaf can cross into the neighbouring band --
        where, with the gate short-circuited off, nothing checks that
        band's cap.
        """
        src = inspect.getsource(GBSpecialBase._cap_drift_gate_setup)
        self.assertIn("cap_drift_gate_edge_leak", src)

        class _Stub:
            cap_divisor = 1
            cap_overlap_frac = 0.0
            cap_drift_gate = True
            _f0_col = 1
            reached_caps = False
            _cap_drift_gate_setup = GBSpecialBase._cap_drift_gate_setup

            @property
            def _cap_leaf_cap(self):
                # reading this proves the divisor short-circuit was passed
                self.reached_caps = True
                return None

        stub = _Stub()
        stub.cap_drift_gate_edge_leak = False
        self.assertIsNone(stub._cap_drift_gate_setup(None))
        self.assertFalse(
            stub.reached_caps,
            "historically the gate short-circuits off at divisor 1 without "
            "overlap, before it ever looks at the caps",
        )

        stub.cap_drift_gate_edge_leak = True
        self.assertIsNone(stub._cap_drift_gate_setup(None))
        self.assertTrue(
            stub.reached_caps,
            "with the edge-leak knob armed the gate must stay live in the "
            "cells == bands configuration",
        )

    def test_recipe_pins_repeats_off_for_non_search_moves(self):
        import lisatools.globalfit.recipe as recipe_mod

        src = inspect.getsource(recipe_mod)
        self.assertIn('"band_unit_repeats": 1', src)
        self.assertIn("_pe_strict or not _gb_mode_search", src)


if __name__ == "__main__":
    unittest.main()
