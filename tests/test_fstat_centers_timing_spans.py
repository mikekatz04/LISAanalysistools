"""Sub-span decomposition of the ``rj_fstat_centers`` [GB_TIMING] stage.

NOT a hunt for missing time -- the bucket is real and already accounts
for itself. v7 production (snapshot 2026-08-28)::

    total=1955.1s  run_proposal=1810.6s
    rj_fstat_centers=1334.874s   <-- 68% of the propose

The ``[FSTAT_CTR ...] unit precompute: ... in 149.75s`` line is emitted
ONCE PER BAND UNIT and the move runs NINE units per propose; the nine
lines sum to ~1,339 s, so the stage closes to 0.2%. (Reading a single
line as the whole precompute manufactures a ~1,185 s phantom hole -- the
mistake this docstring exists to prevent.) Cross-checked twice: the other
``_run_rj_step`` marks bound the per-round centre chain at <= 3.077 s,
and the nine unit row counts sum to exactly ``picked_sources`` =
4,546,846.

So ~99.8% of the stage is ONE path -- ``_precompute_fstat_centers`` ->
``_fstat_ctr_compute`` -- at 4.55 M rows/propose, ~0.293 ms/row, ~97% of
them dead birth slots.

WHAT THESE SPANS ARE FOR: splitting the INTERIOR of that 1,331 s, which
nobody has measured. ``fstat_nm_transform`` vs ``fstat_nm_lanes`` (with
``fstat_nm_h2d`` / ``fstat_nm_lane_score``) vs ``fstat_nm_invert`` vs
``fstat_ctr_map`` decides whether the lever is a cheaper kernel or fewer
rows. The surrounding stages (select / census / audit / pack, and the
small per-round chain) are named so the split can be read without
subtraction.

The sync caveat still applies to the SHARES (not to the total): the
default timer does not synchronize (``GB_PROP_TIMING_SYNC=0`` in v7), so
a phase containing a sync point absorbs queued device work from earlier
phases -- ``fill_indmap_data`` measured 598 s and was really 45 s.
Splitting an interior is exactly where that bites, so the sub-spans must
be honest in BOTH modes and the sync-on mode must be reachable without a
production cost.

The invariants:

* every new stage is NESTED (inside ``run_proposal``), so none of them may
  join ``_ProposeTimer.report``'s ``_TOP`` list -- that would double-count
  seconds already carried by ``run_proposal``. They still print, because
  ``report`` prints every stage it holds.
* the two ``_mark("rj_fstat_centers")`` checkpoints SURVIVE, so the
  headline number stays comparable with every earlier log.
* instrumentation adds no device sync on the default path and does not
  touch the RNG draw order.
"""

from __future__ import annotations

import inspect
import types
import unittest

import numpy as np


# Every sub-stage this decomposition introduces. All NESTED.
_NEW_STAGES = (
    # unit-open precompute (_precompute_fstat_centers)
    "fstat_ctr_select",
    "fstat_ctr_solve",
    "fstat_ctr_census",
    "fstat_ctr_audit",
    "fstat_ctr_pack",
    # shared per-row scorer (_fstat_dist_centers / _fstat_NM /
    # _fstat_ctr_compute), whoever calls it
    "fstat_nm_transform",
    "fstat_nm_lanes",
    "fstat_nm_routed",
    "fstat_nm_invert",
    "fstat_ctr_map",
    "fstat_nm_lane_build",
    # emitted from gbbands' call_NM, nested inside fstat_nm_lanes
    "fstat_nm_h2d",
    "fstat_nm_lane_score",
    "fstat_ctr_miss_fallback",
    # per-pick-round centre chain (_run_rj_step)
    "rj_ctr_keep_gate",
    "rj_ctr_birth_lookup",
    "rj_ctr_birth_draw",
    "rj_ctr_death_lookup",
    "rj_ctr_death_dens",
)


def _mod():
    from lisatools.globalfit.moves import gbspecialstretch

    return gbspecialstretch


class _CountingSync:
    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1


def _stub_move(timer=None, n=64, cap_mask=None, alive=None):
    """A duck-typed stand-in carrying the real precompute methods."""
    gs = _mod()

    ids = np.arange(n, dtype=np.int64)
    if alive is None:
        alive = np.zeros(n, dtype=bool)
        alive[: n // 2] = True

    subset = types.SimpleNamespace(
        inds_main_band_sorter=ids, inds=alive)
    sorter = types.SimpleNamespace(coords=np.zeros((n, 9)))

    calls = []

    def _compute(model, params, smear=None):
        calls.append(("compute", int(params.shape[0])))
        m = int(params.shape[0])
        return tuple(np.zeros(m) for _ in range(6))

    def _audit(*a, **kw):
        calls.append(("audit", 0))

    stub = types.SimpleNamespace(
        xp=np,
        name="rj_fstat_search",
        _prop_timer=timer,
        _rj_at_cap_mask=cap_mask,
        _fstat_ctr_fallback_rows=0,
        _fstat_ctr_compute=_compute,
        _fstat_ctr_audit=_audit,
        _unit_cache_smear=lambda: 1.5,
        calls=calls,
    )
    stub._precompute = types.MethodType(
        gs.GBSpecialBase._precompute_fstat_centers, stub)
    return stub, subset, sorter


class NestedAccountingTest(unittest.TestCase):
    """New stages print but never join the tracked total."""

    def test_new_stages_stay_untracked(self):
        gs = _mod()
        tm = gs._ProposeTimer()
        tm.add("run_proposal", 1000.0)
        for name in _NEW_STAGES:
            tm.add(name, 7.0)
        line = tm.report(total=1000.0)
        self.assertIn("tracked=1000.000s", line)
        self.assertIn("untracked=0.000s", line)

    def test_new_stages_are_visible_in_the_line(self):
        gs = _mod()
        tm = gs._ProposeTimer()
        for name in _NEW_STAGES:
            tm.add(name, 1.5)
        line = tm.report(total=100.0)
        for name in _NEW_STAGES:
            self.assertIn(f"{name}=1.500s", line)

    def test_stage_registry_is_exported_and_matches(self):
        # A single source of truth the report() comment and the tests read,
        # so a future stage cannot be added to the code and forgotten here.
        gs = _mod()
        self.assertEqual(
            tuple(gs._FSTAT_CTR_SUBSTAGES), tuple(_NEW_STAGES))

    def test_registry_names_are_disjoint_from_top(self):
        gs = _mod()
        tm = gs._ProposeTimer()
        top = inspect.signature(tm.report).parameters
        self.assertIn("top", top)
        # report()'s default _TOP must not contain any nested stage.
        line = gs._ProposeTimer().report(total=0.0)
        self.assertIn("tracked=0.000s", line)
        for name in gs._FSTAT_CTR_SUBSTAGES:
            tm2 = gs._ProposeTimer()
            tm2.add(name, 5.0)
            self.assertIn("tracked=0.000s", tm2.report(total=5.0))


class PrecomputeSubSpanTest(unittest.TestCase):
    """``_precompute_fstat_centers`` decomposes into its real phases."""

    def test_records_every_precompute_substage(self):
        gs = _mod()
        tm = gs._ProposeTimer()
        stub, subset, sorter = _stub_move(timer=tm)
        out = stub._precompute(object(), sorter, subset)
        self.assertIsNotNone(out)
        for name in ("fstat_ctr_select", "fstat_ctr_solve",
                     "fstat_ctr_census", "fstat_ctr_audit",
                     "fstat_ctr_pack"):
            self.assertIn(name, tm.stages, f"missing sub-span {name}")

    def test_audit_is_timed_separately_from_the_solve(self):
        # The audit runs AFTER the census log line, i.e. inside the
        # rj_fstat_centers span but OUTSIDE the census's "in %.2fs"
        # number -- so it can only be costed by its own span.
        gs = _mod()
        tm = gs._ProposeTimer()
        stub, subset, sorter = _stub_move(timer=tm)

        def _slow_audit(*a, **kw):
            for _ in range(20000):
                pass

        stub._fstat_ctr_audit = _slow_audit
        stub._precompute(object(), sorter, subset)
        self.assertGreater(tm.stages["fstat_ctr_audit"],
                           tm.stages["fstat_ctr_pack"])

    def test_row_counts_are_reported(self):
        gs = _mod()
        tm = gs._ProposeTimer()
        stub, subset, sorter = _stub_move(timer=tm, n=64)
        stub._precompute(object(), sorter, subset)
        self.assertEqual(tm.counts.get("fstat_ctr_units"), 1)
        self.assertEqual(tm.counts.get("fstat_ctr_rows"), 64)

    def test_no_timer_is_behaviour_identical(self):
        gs = _mod()
        tm = gs._ProposeTimer()
        a_stub, a_sub, a_sort = _stub_move(timer=tm)
        b_stub, b_sub, b_sort = _stub_move(timer=None)
        a = a_stub._precompute(object(), a_sort, a_sub)
        b = b_stub._precompute(object(), b_sort, b_sub)
        self.assertEqual(a_stub.calls, b_stub.calls)
        self.assertEqual(sorted(a.keys()), sorted(b.keys()))
        np.testing.assert_array_equal(a["ids"], b["ids"])

    def test_sync_fires_at_every_substage_boundary(self):
        # GB_PROP_TIMING_SYNC=1 -> the operator's ONE trustworthy propose:
        # every sub-span boundary drains the device, so each number is that
        # phase's own kernel time rather than a drain of earlier phases.
        gs = _mod()
        sync = _CountingSync()
        tm = gs._ProposeTimer(sync_fn=sync)
        stub, subset, sorter = _stub_move(timer=tm)
        stub._precompute(object(), sorter, subset)
        # 5 sub-spans, opened and closed.
        self.assertGreaterEqual(sync.n, 10)

    def test_default_path_adds_no_sync(self):
        gs = _mod()
        tm = gs._ProposeTimer()          # sync_fn=None -- production
        self.assertIsNone(tm._sync)
        stub, subset, sorter = _stub_move(timer=tm)
        stub._precompute(object(), sorter, subset)
        self.assertIn("fstat_ctr_solve", tm.stages)

    def test_empty_unit_returns_none_without_spans(self):
        gs = _mod()
        tm = gs._ProposeTimer()
        stub, subset, sorter = _stub_move(timer=tm, n=0)
        self.assertIsNone(stub._precompute(object(), sorter, subset))


class ScorerSubSpanTest(unittest.TestCase):
    """The per-row F-stat scorer decomposes wherever it is called from."""

    def _scorer_stub(self, tm, lanes=None, n=8):
        gs = _mod()
        stub = types.SimpleNamespace(
            xp=np,
            name="rj_fstat_search",
            _prop_timer=tm,
            _fstat_nm_lanes=lanes,
            transform_fn=types.SimpleNamespace(
                both_transforms=lambda p, xp=None: np.asarray(p)),
        )
        stub._fstat_NM = lambda model, p, w: (
            np.zeros((int(p.shape[0]), 4)),
            np.tile(
                np.array([1.0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 1.0]),
                (int(p.shape[0]), 1)),
        )
        stub._fstat_dist_centers = types.MethodType(
            gs.GBSpecialBase._fstat_dist_centers, stub)
        return stub

    def test_dist_centers_splits_transform_and_inversion(self):
        gs = _mod()
        tm = gs._ProposeTimer()
        stub = self._scorer_stub(tm)
        stub._fstat_dist_centers(object(), np.zeros((8, 9)), 0)
        self.assertIn("fstat_nm_transform", tm.stages)
        self.assertIn("fstat_nm_invert", tm.stages)

    def test_nm_lane_path_is_named_separately(self):
        gs = _mod()
        tm = gs._ProposeTimer()
        seen = []

        def _lane_call(p):
            seen.append(int(p.shape[0]))
            n = int(p.shape[0])
            return np.zeros((n, 4)), np.zeros((n, 10))

        stub = types.SimpleNamespace(
            xp=np, name="m", _prop_timer=tm,
            _fstat_nm_lanes=(3, _lane_call))
        stub._fstat_NM = types.MethodType(gs.GBSpecialBase._fstat_NM, stub)
        stub._fstat_NM(object(), np.zeros((5, 9)), 3)
        self.assertEqual(seen, [5])
        self.assertIn("fstat_nm_lanes", tm.stages)
        self.assertNotIn("fstat_nm_routed", tm.stages)
        self.assertEqual(tm.counts.get("fstat_nm_rows"), 5)


class RjRoundCentreChainTest(unittest.TestCase):
    """The per-pick-round centre chain in ``_run_rj_step``.

    Structural, not functional: ``_run_rj_step`` needs a live buffer, a
    GPU sorter and the likelihood engine, so the marks are pinned by
    reading the source (the established idiom in
    ``tests/test_band_unit_scan_order.py``).
    """

    def setUp(self):
        gs = _mod()
        self.src = inspect.getsource(gs.GBSpecialBase._run_rj_step)

    def test_every_round_substage_is_marked(self):
        for name in ("rj_ctr_keep_gate", "rj_ctr_birth_lookup",
                     "rj_ctr_birth_draw", "rj_ctr_death_lookup",
                     "rj_ctr_death_dens"):
            self.assertIn(f'"{name}"', self.src, f"missing mark {name}")

    def test_headline_mark_survives_for_comparability(self):
        # Nested marks must NOT replace the parent checkpoint: every log
        # since 2026-08-15 quotes rj_fstat_centers.
        self.assertEqual(self.src.count('_mark("rj_fstat_centers")'), 2)

    def test_keep_gate_closes_before_the_birth_lookup(self):
        # bool(keep.any()) + the boolean fancy-index gathers are the FIRST
        # syncs after rj_prior_gate: under GB_PROP_TIMING_SYNC=0 they drain
        # the prior gate's queued kernels. They must be costed on their own
        # so that drain is not billed to the centre lookup.
        i_gate = self.src.index('"rj_ctr_keep_gate"')
        i_look = self.src.index('"rj_ctr_birth_lookup"')
        self.assertLess(i_gate, i_look)

    def test_birth_lookup_closes_before_the_draw(self):
        i_look = self.src.rindex('"rj_ctr_birth_lookup"')
        i_draw = self.src.index('"rj_ctr_birth_draw"')
        i_rng = self.src.index("_truncnorm_std_draw(len(birth_k)")
        self.assertLess(i_look, i_rng)
        self.assertLess(i_rng, i_draw)

    def test_rng_call_sites_are_unchanged(self):
        # Bit-identical sampling: instrumentation may not add, remove or
        # reorder a draw. These are the RJ step's only RNG consumers.
        self.assertEqual(self.src.count("_truncnorm_std_draw("), 1)
        self.assertEqual(self.src.count("cp.random.randn("), 1)
        self.assertEqual(self.src.count("cp.random.rand("), 1)


class LaneAdapterTimerTest(unittest.TestCase):
    """``make_fstat_nm_lanes`` must be able to report its own phases."""

    def test_accepts_an_explicit_timer(self):
        from lisatools.globalfit.moves.gbbands import _RoutedBandEngine

        sig = inspect.signature(_RoutedBandEngine.make_fstat_nm_lanes)
        self.assertIn("timer", sig.parameters)
        self.assertIsNone(sig.parameters["timer"].default)

    def test_h2d_and_lane_score_are_named(self):
        from lisatools.globalfit.moves.gbbands import _RoutedBandEngine

        src = inspect.getsource(_RoutedBandEngine.make_fstat_nm_lanes)
        self.assertIn('"fstat_nm_h2d"', src)
        self.assertIn('"fstat_nm_lane_score"', src)


class SyncModeTest(unittest.TestCase):
    """``GB_PROP_TIMING_SYNC=all`` drains EVERY run device.

    ``deviceSynchronize`` drains only the CURRENT device. With the
    multi-device F-stat NM lanes armed (``GB_FSTAT_NM_MULTIDEV=1``, the v7
    default) a current-device-only sync leaves the other GPU's queue
    outstanding, so even the sync-on decomposition would misattribute.
    """

    def test_none_when_disabled(self):
        gs = _mod()
        self.assertIsNone(gs._prop_timer_sync_fn(object(), [0, 1], "0"))

    def test_current_device_mode(self):
        gs = _mod()
        calls = []
        xp = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                runtime=types.SimpleNamespace(
                    deviceSynchronize=lambda: calls.append("cur")),
                Device=None,
            )
        )
        fn = gs._prop_timer_sync_fn(xp, [0, 1], "1")
        fn()
        self.assertEqual(calls, ["cur"])

    def test_all_device_mode_visits_every_gpu(self):
        gs = _mod()
        calls = []

        class _Dev:
            def __init__(self, i):
                self.i = i

            def __enter__(self):
                calls.append(("enter", self.i))
                return self

            def __exit__(self, *a):
                calls.append(("exit", self.i))
                return False

        xp = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                runtime=types.SimpleNamespace(
                    deviceSynchronize=lambda: calls.append("sync")),
                Device=_Dev,
            )
        )
        fn = gs._prop_timer_sync_fn(xp, [0, 1], "all")
        fn()
        self.assertEqual(
            calls,
            [("enter", 0), "sync", ("exit", 0),
             ("enter", 1), "sync", ("exit", 1)],
        )

    def test_all_mode_without_gpus_falls_back(self):
        gs = _mod()
        calls = []
        xp = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                runtime=types.SimpleNamespace(
                    deviceSynchronize=lambda: calls.append("cur")),
                Device=None,
            )
        )
        fn = gs._prop_timer_sync_fn(xp, [], "all")
        fn()
        self.assertEqual(calls, ["cur"])


if __name__ == "__main__":
    unittest.main()
