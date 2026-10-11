"""Which residual the monitor's "whitened residual per GB band" panels use
for each cold walker (2026-10-11).

The panels compare the max-lnL (cyan) and the min-lnL (amber) cold walker.
The run's residual snapshot used to hold the max-lnL walker alone, and the
page rebuilt every OTHER walker as data minus its GB + VGB templates --
every MBH / EMRI / SOBBH left in. On the 9mo page (row 89) that read as a
5-7x amber excess below 1.5 mHz for a walker only ~800 nats behind.

``walker_residual_x`` now picks, best first: the run's own per-walker
residual; an old single-walker snapshot with this walker's GB + VGB swapped
in; GB + VGB subtraction only (no snapshot). The generator is a script, so
its helpers are compiled by name, as ``tests/test_monitor_band_ll.py`` does.
Also here: the valve panel's tolerance comes from the run's [GB_GATE] line,
not the page builder's env.
"""
import unittest

import numpy as np

from lisatools.globalfit.monitor import generator_path
from lisatools.globalfit.monitor._short import _generator_functions

N = 6


def _c(seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=N) + 1j * rng.normal(size=N)


class _Calls:
    """``tmpl_x`` / ``snap_fd_x`` stand-ins that record what was asked."""

    def __init__(self):
        self.tmpl = {v: _c(100 + v) for v in range(4)}
        self.t_calls, self.fd_calls = [], []

    def tmpl_x(self, v):
        self.t_calls.append(int(v))
        return self.tmpl[int(v)].copy()

    def snap_fd_x(self, arr):
        # stand-in transform: any fixed map of the WDM array will do
        self.fd_calls.append(arr)
        return np.asarray(arr)[0, 0].astype(complex) * (2.0 + 1.0j)


class WalkerResidualTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = staticmethod(_generator_functions(
            {"np": np}, names=("walker_residual_x",))["walker_residual_x"])

    def setUp(self):
        self.c = _Calls()
        self.data_x = _c(1)
        self.page_x = _c(2)     # the page's resid_fd[0]

    def _call(self, w, wbest, snap):
        return self.f(w, wbest, snap, self.page_x, self.c.tmpl_x,
                      self.c.snap_fd_x, self.data_x)

    # -- no snapshot: GB + VGB subtraction only --------------------------------
    def test_no_snapshot_subtracts_gb_vgb_only_and_says_so(self):
        r, kind, label = self._call(2, 0, None)
        np.testing.assert_allclose(r, self.data_x - self.c.tmpl[2])
        self.assertEqual(kind, "gbvgb")
        self.assertEqual(label, "GB+VGB subtraction only (MBH / EMRI / SOBBH not "
                                "subtracted)")

    def test_no_snapshot_best_walker_reuses_the_page_residual(self):
        r, kind, _ = self._call(0, 0, None)
        self.assertIs(r, self.page_x)
        self.assertEqual(kind, "gbvgb")
        self.assertEqual(self.c.t_calls, [])

    # -- OLD single-walker snapshot: swap this walker's GB + VGB in ----------
    def test_old_snapshot_swaps_this_walkers_gb_vgb_into_the_snapshot_residual(self):
        snap = {"walker": np.int64(1), "residual": np.zeros((3, 1, N), np.float32)}
        r, kind, label = self._call(0, 1, snap)
        # r_min = r_s + T(s) - T(w): only the GB + VGB part is this walker's
        np.testing.assert_allclose(r, self.page_x + self.c.tmpl[1] - self.c.tmpl[0])
        self.assertEqual(kind, "swap")
        self.assertEqual(label, "snapshot residual of walker 1 with this walker's "
                                "GB+VGB templates swapped in (other classes from "
                                "walker 1)")
        self.assertEqual(sorted(self.c.t_calls), [0, 1])
        self.assertEqual(self.c.fd_calls, [])
        # and the page's own residual is untouched by the in-place subtract
        np.testing.assert_allclose(self.page_x, _c(2))

    def test_old_snapshot_its_own_walker_is_the_runs_residual(self):
        snap = {"walker": 1, "residual": np.zeros((3, 1, N), np.float32)}
        r, kind, label = self._call(1, 0, snap)
        self.assertIs(r, self.page_x)
        self.assertEqual((kind, label), ("run", "run's own residual"))
        self.assertEqual((self.c.t_calls, self.c.fd_calls), ([], []))

    def test_old_snapshot_of_another_walker_swaps_for_the_best_one_too(self):
        # the snapshot's max-lnL walker (2) need not be the store's last-row
        # best (0): then the cyan curve is a swap as well, and says so
        snap = {"walker": 2, "residual": np.zeros((3, 1, N), np.float32)}
        r, kind, _ = self._call(0, 0, snap)
        self.assertEqual(kind, "swap")
        np.testing.assert_allclose(r, self.page_x + self.c.tmpl[2] - self.c.tmpl[0])

    # -- NEW per-walker snapshot: the run's own residual for every walker ----
    def test_per_walker_snapshot_uses_the_runs_own_residual_for_each_walker(self):
        rs = np.stack([np.full((3, 1, N), 10.0 + k, np.float32) for k in range(2)])
        # the page keeps a SUBSET (max- and min-lnL), in walker order
        snap = {"walker": 1, "residual": rs[0], "residuals": rs, "walkers": [1, 3]}
        r, kind, label = self._call(3, 1, snap)
        np.testing.assert_allclose(r, 11.0 * (2.0 + 1.0j) * np.ones(N))
        self.assertEqual((kind, label), ("run", "run's own residual"))
        self.assertEqual(self.c.t_calls, [])            # no template rebuilt
        self.assertEqual(len(self.c.fd_calls), 1)

    def test_per_walker_snapshot_without_this_walker_falls_back_to_the_swap(self):
        rs = np.stack([np.full((3, 1, N), 10.0, np.float32)])
        snap = {"walker": 1, "residual": rs[0], "residuals": rs, "walkers": [1]}
        r, kind, _ = self._call(2, 1, snap)
        self.assertEqual(kind, "swap")
        np.testing.assert_allclose(r, self.page_x + self.c.tmpl[1] - self.c.tmpl[2])


class SnapshotToFdTest(unittest.TestCase):
    """The X-only conversion the per-walker path uses equals channel X of the
    3-channel one the page has always used (a 1-channel WDMSignal)."""

    def test_x_only_matches_channel_x_of_the_full_conversion(self):
        from lisatools.domains import FDSettings, WDMSettings

        ns = _generator_functions({"np": np}, names=("snapshot_wdm_to_fd",))
        nf, nt, dt = 16, 32, 2.0
        full = WDMSettings(nf, nt, dt, force_backend="cpu")
        # active band AND edge-cropped span, as the run's grid has
        wdm = WDMSettings(nf, nt, dt, min_freq=3.0 / (2 * nf * dt),
                          max_freq=11.0 / (2 * nf * dt), min_time=4 * nf * dt,
                          max_time=(nt - 4) * nf * dt, force_backend="cpu")
        self.assertLess(int(wdm.Nt_active), nt)
        self.assertEqual((int(full.Nf), int(full.Nt)), (nf, nt))
        n = nf * nt
        fds = FDSettings(n // 2 + 1, 1.0 / (n * dt), force_backend="cpu")
        rng = np.random.default_rng(0)
        arr = rng.normal(size=(3, int(wdm.Nf_active), int(wdm.Nt_active))).astype(np.float32)
        three = ns["snapshot_wdm_to_fd"](arr, wdm, full, fds)
        x_only = ns["snapshot_wdm_to_fd"](arr, wdm, full, fds, channels=slice(0, 1))
        self.assertEqual(three.shape, (3, fds.N))
        self.assertEqual(x_only.shape, (1, fds.N))
        np.testing.assert_allclose(x_only[0], three[0], rtol=0, atol=1e-12 * np.abs(three).max())
        self.assertGreater(np.abs(three[0]).max(), 0.0)


class GateToleranceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = staticmethod(_generator_functions(
            {"np": np}, names=("gb_gate_valve_settings",))["gb_gate_valve_settings"])

    LINE = ("2026-10-10 03:01:02 [GB_GATE {mv}] occupied 812 | shut 120 (14.8%) | "
            "active 692 -- streak p50 1 p90 2 of {ci}; resets/pair mean 3.10 max 9; "
            "chronic 400 young 200 fresh 92 | max-minus-now gain p50 5.21 p90 30.10 "
            "(tol {tol}) | peak: 812 of 812 occupied cells observed this cycle "
            "(4 walker(s) reporting)\n")

    def test_the_newest_gate_line_wins_over_the_local_env(self):
        log = (self.LINE.format(mv="rj_prior", ci=3, tol="4.00")
               + "other line\n"
               + self.LINE.format(mv="rj_fstat_search", ci=5, tol="20.00"))
        env = {"GB_SEARCH_BAND_SHUTOFF_LL_TOL": "4.0"}
        self.assertEqual(self.f(log, env), (20.0, 5.0, True, True))

    def test_without_a_gate_line_the_env_is_used_and_flagged(self):
        self.assertEqual(self.f("", {}), (4.0, 3.0, False, False))
        self.assertEqual(self.f("[GB_GATE x] RJ shutoff floor 1.2 mHz in force\n",
                                {"GB_SEARCH_BAND_SHUTOFF_LL_TOL": "7"}),
                         (7.0, 3.0, False, False))


class PageWiringTest(unittest.TestCase):
    """The page block calls the helpers; the old two-way rule is gone."""

    @classmethod
    def setUpClass(cls):
        with open(generator_path(), encoding="utf-8") as fh:
            cls.src = fh.read()
        cls.block = cls.src[cls.src.index("FOREGROUND DIAGNOSIS: the whitened residual"):
                            cls.src.index("FOREGROUND RATCHET: the timeline")]

    def test_every_shown_walker_goes_through_walker_residual_x(self):
        self.assertIn("walker_residual_x(\n                _w, WBEST, SNAP, resid_fd[0], "
                      "_gbvgb_X, _snap_fd_x, data_fd[0])", self.block)
        self.assertNotIn("_walker_resid_X", self.src)
        # the construction is said on all three figures and in the prose
        self.assertEqual(self.block.count("residuals -- {_wsrc_t}"), 3)
        self.assertEqual(self.src.count("{FG_RESID_SRC}"), 2)
        self.assertIn('"<em>Caveats.</em> (1) " + _F["src_txt"]', self.src)

    def test_the_valve_panel_reads_the_runs_tolerance(self):
        self.assertIn("gb_gate_valve_settings(log_text, os.environ)", self.src)
        self.assertNotIn('_tol = float(os.environ.get("GB_SEARCH_BAND_SHUTOFF_LL_TOL"',
                         self.src)
        # defined BEFORE the panel that uses it: the generator runs top-down
        self.assertLess(self.src.index("def gb_gate_valve_settings("),
                        self.src.index("gb_gate_valve_settings(log_text, os.environ)"))

    def test_the_snapshot_reader_is_the_shared_helper(self):
        self.assertIn("SNAP = load_residual_snapshot(", self.src)
        self.assertLess(self.src.index("def load_residual_snapshot("),
                        self.src.index("SNAP = load_residual_snapshot("))
        self.assertLess(self.src.index("def walker_residual_x("),
                        self.src.index("walker_residual_x(\n"))


if __name__ == "__main__":
    unittest.main()
