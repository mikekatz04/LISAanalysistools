"""MBH / EMRI / SOBBH default per-leaf ladders and their resume handling.

User ruling 2026-10-13: "the default ladders for non-RJ sources should be
from make_ladder. not from the 1/1.2^k setup." Each branch Setup builds
``make_ladder(ndim, ntemps={BRANCH}_NTEMPS)`` (Tmax unset: the ~25 %
swap-acceptance geometric spacing of an ndim Gaussian) when no explicit
``betas`` is set; an explicit ``betas`` wins.

Resume: ``SingleSourcePEBuilder.build`` used to RE-TILE the configured ladder
over every leaf at every build, so each resume silently reset the stored,
adapted per-leaf ladders (the 9mo v9 store reads exactly 1/1.2^k on every
leaf of all three branches at row 113, job 757's first row, after rows that
had drifted by up to ~2e-3). Now a resumed store keeps its stored
``betas_all``; ``{BRANCH}_LADDER_RESET_ON_RESUME=1`` replaces them with the
configured ladder (logged leaf by leaf). The rung count stays the store's
(``resume_ladder_wins``), so the shapes reconcile with run.py's checks.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np
from eryn.moves.tempering import make_ladder

#: every env knob these tests read: cleared so a developer shell cannot leak in
_KNOBS = tuple(
    f"{b}_{k}"
    for b in ("EMRI", "MBH", "SOBBH")
    for k in ("NTEMPS", "LADDER_RESET_ON_RESUME", "LADDER_REANCHOR_NATS",
              "EIGEN_JUMP_MIX", "EIGEN_JUMP_FACTOR", "INNER_MOVE_KIND")
)

#: launcher ntemps (submit_gf_{6mo_v9_4gpu,1yr_v9,9mo_v9}.sh: 8 / 8 / 8) and
#: the _lite presets' (variants/lite.py: 2)
LAUNCHER_NTEMPS = 8
LITE_NTEMPS = 2
TOBS, DT = 31536000.0, 10.0


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in _KNOBS}
    env.update(extra)
    return mock.patch.dict(os.environ, env, clear=True)


def _settings(branch, **kw):
    from lisatools.globalfit.stock.erebor import source_runtime as sr

    cls = {"emri": sr.SourceEMRISettings, "mbh": sr.SourceMBHSettings,
           "sobbh": sr.SourceSOBBHSettings}[branch]
    if branch == "emri":
        # prepare_emri_branch turns the empty-list limits into None (wide priors)
        for lims in ("logm1_lims", "m2_lims", "a_lims", "p0_lims", "e0_lims"):
            kw.setdefault(lims, None)
    return cls(**kw)


def _setup_cls(branch):
    from lisatools.globalfit.stock.erebor.emri import EMRISetup
    from lisatools.globalfit.stock.erebor.mbh import MBHSetup
    from lisatools.globalfit.stock.erebor.sobbh import SOBBHSetup

    return {"emri": EMRISetup, "mbh": MBHSetup, "sobbh": SOBBHSetup}[branch]


class DefaultLadderTest(unittest.TestCase):
    """The Setups' default ``betas`` is make_ladder(ndim, ntemps)."""

    NDIM = {"emri": 12, "mbh": 11, "sobbh": 11}

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _setup(self, branch, **kw):
        with _clean_env(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = _settings(branch, Tobs=TOBS, dt=DT, log_dir=self.tmp, inner_moves=[], **kw)
            return _setup_cls(branch)(s)

    def test_launcher_and_lite_ntemps(self):
        for branch, ndim in self.NDIM.items():
            for nt in (LAUNCHER_NTEMPS, LITE_NTEMPS):
                with self.subTest(branch=branch, ntemps=nt):
                    setup = self._setup(branch, ntemps=nt)
                    np.testing.assert_array_equal(setup.betas, make_ladder(ndim, ntemps=nt))
                    # not the retired 1/1.2^k
                    self.assertFalse(np.allclose(setup.betas, 1 / 1.2 ** np.arange(nt)))

    def test_the_stock_ndim_is_the_sampling_basis(self):
        """The ndim fed to make_ladder is the branch's real sampling ndim."""
        for branch, ndim in self.NDIM.items():
            with self.subTest(branch=branch):
                setup = self._setup(branch, ntemps=4)
                self.assertEqual(int(setup.ndim), ndim)
                self.assertEqual(len(setup.transform.input_basis), ndim)

    def test_unset_ndim_falls_back_to_the_input_basis(self):
        from lisatools.globalfit.stock.erebor.mbh import MBHSettings, MBHSetup

        with _clean_env(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setup = MBHSetup(MBHSettings(Tobs=TOBS, dt=DT, log_dir=self.tmp, ntemps=8))
        self.assertIsNone(setup.ndim)
        np.testing.assert_array_equal(setup.betas, make_ladder(11, ntemps=8))

    def test_the_env_ntemps_sizes_it(self):
        with _clean_env(EMRI_NTEMPS="5"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setup = _setup_cls("emri")(
                _settings("emri", Tobs=TOBS, dt=DT, log_dir=self.tmp, inner_moves=[]))
        np.testing.assert_array_equal(setup.betas, make_ladder(12, ntemps=5))

    def test_an_explicit_betas_wins(self):
        mine = np.array([1.0, 0.7, 0.3])
        for branch in self.NDIM:
            with self.subTest(branch=branch):
                setup = self._setup(branch, ntemps=8, betas=mine.copy())
                np.testing.assert_array_equal(setup.betas, mine)

    def test_the_ladder_is_logged(self):
        for branch in self.NDIM:
            with self.subTest(branch=branch):
                with self.assertLogs(f"{branch.upper()}Setup", level="INFO") as cm:
                    self._setup(branch, ntemps=3)
                self.assertTrue(any("Using betas" in line for line in cm.output), cm.output)


class LadderKnobDefaultsTest(unittest.TestCase):
    def test_defaults(self):
        with _clean_env():
            for branch in ("emri", "mbh", "sobbh"):
                with self.subTest(branch=branch):
                    s = _settings(branch)
                    self.assertIs(s.ladder_reset_on_resume, False)
                    self.assertEqual(s.ladder_reanchor_nats, 200.0)
                    self.assertEqual(s.eigen_jump_mix, 0.0)
                    self.assertEqual(s.eigen_jump_factor, 3.0)

    def test_env_knobs_follow_the_attribute_rule(self):
        for branch in ("emri", "mbh", "sobbh"):
            B = branch.upper()
            env = {f"{B}_LADDER_RESET_ON_RESUME": "1", f"{B}_LADDER_REANCHOR_NATS": "75",
                   f"{B}_EIGEN_JUMP_MIX": "0.25", f"{B}_EIGEN_JUMP_FACTOR": "5"}
            with self.subTest(branch=branch), _clean_env(**env):
                s = _settings(branch)
                self.assertIs(s.ladder_reset_on_resume, True)
                self.assertEqual(
                    (s.ladder_reanchor_nats, s.eigen_jump_mix, s.eigen_jump_factor),
                    (75.0, 0.25, 5.0))

    def test_settings_survive_deepcopy_and_pickle(self):
        import copy
        import pickle

        with _clean_env(EMRI_EIGEN_JUMP_MIX="0.1"):
            s = _settings("emri", ladder_reset_on_resume=True)
        t = pickle.loads(pickle.dumps(copy.deepcopy(s)))
        self.assertIs(t.ladder_reset_on_resume, True)
        self.assertEqual(t.eigen_jump_mix, 0.1)


class LadderTmaxCapTest(unittest.TestCase):
    """``{BRANCH}_LADDER_TMAX`` caps the default ladder's top temperature:
    make_ladder's unique geometric spacing from 1 to Tmax over ntemps rungs.
    Unset (the default) is the ruling's uncapped ~25 % spacing, unchanged."""

    NDIM = {"emri": 12, "mbh": 11, "sobbh": 11}

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _setup(self, branch, env=None, **kw):
        with _clean_env(**(env or {})), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = _settings(branch, Tobs=TOBS, dt=DT, log_dir=self.tmp, inner_moves=[], **kw)
            return s, _setup_cls(branch)(s)

    def test_unset_is_the_uncapped_ladder(self):
        for branch, ndim in self.NDIM.items():
            with self.subTest(branch=branch):
                s, setup = self._setup(branch, ntemps=8)
                self.assertIsNone(s.ladder_tmax)
                np.testing.assert_array_equal(setup.betas, make_ladder(ndim, ntemps=8))

    def test_the_cap_is_the_geometric_spacing_to_tmax(self):
        for branch, ndim in self.NDIM.items():
            B = branch.upper()
            with self.subTest(branch=branch):
                s, setup = self._setup(branch, env={f"{B}_LADDER_TMAX": "10"}, ntemps=8)
                self.assertEqual(s.ladder_tmax, 10.0)
                want = make_ladder(ndim, ntemps=8, Tmax=10.0)
                np.testing.assert_array_equal(setup.betas, want)
                self.assertEqual(len(setup.betas), 8)
                self.assertAlmostEqual(float(setup.betas[0]), 1.0)
                self.assertAlmostEqual(1.0 / float(setup.betas[-1]), 10.0, places=9)
                # colder than the uncapped ladder on every rung above the cold one
                self.assertTrue(np.all(setup.betas[1:] > make_ladder(ndim, ntemps=8)[1:]))

    def test_an_explicit_betas_still_wins(self):
        mine = np.array([1.0, 0.5, 0.25])
        _, setup = self._setup("emri", env={"EMRI_LADDER_TMAX": "10"}, ntemps=3, betas=mine)
        np.testing.assert_array_equal(setup.betas, mine)

    def test_a_cap_at_or_below_one_is_refused(self):
        from lisatools.globalfit.stock.erebor.common import default_source_ladder

        for bad in (1.0, 0.5, -3.0):
            with self.assertRaises(ValueError):
                default_source_ladder(12, None, 8, tmax=bad)


# ---------------------------------------------------------------------------
# the resume path through the REAL builder (fake move class, no waveforms)
# ---------------------------------------------------------------------------

class _RecordingMove:
    """Stands in for the add/remove move: records what the builder hands it."""

    fanout_flat_body = True

    def __init__(self, branch_name, coords_shape, *args, **kwargs):
        self.branch_name = branch_name
        self.coords_shape = coords_shape
        self.kwargs = kwargs

    def install_walker_fanout(self, curr):
        self.installed = True


NW, NLEAVES, NDIM = 4, 3, 12


def _builder(**kw):
    from lisatools.globalfit.recipe import SingleSourcePEBuilder

    class _B(SingleSourcePEBuilder):
        branch_name = "emri"
        move_class = _RecordingMove
        use_dcga = False

    return _B(wave_gen=lambda *a, **k: None, **kw)


def _info(configured, reset=False, reanchor=200.0):
    return SimpleNamespace(
        betas=None if configured is None else np.asarray(configured, dtype=float),
        ntemps=None if configured is None else len(configured),
        ndim=NDIM, nleaves_max=NLEAVES, waveform_kwargs={}, num_prop_repeats=2,
        transform=None, inner_moves=[], eigen_refresh_every=None,
        eigen_table_scope=None, info_matrix_gen=None,
        ladder_reset_on_resume=reset, ladder_reanchor_nats=reanchor,
    )


def _sub(ntemps, stored=None):
    """A loaded sub-state: ``stored`` = the store's betas_all (None = fresh)."""
    from lisatools.globalfit.state import EMRIState

    sub = EMRIState(None, betas_all=None if stored is None else np.array(stored, copy=True))
    if stored is None:
        sub._set_leaf_count(NLEAVES)
    sub.initialize_tempered(ntemps, NW, NLEAVES, NDIM)
    return sub


def _adapted(ntemps, seed=3):
    """Per-leaf adapted ladders: DIFFERENT per leaf, cold rung 1, decreasing."""
    rng = np.random.default_rng(seed)
    base = make_ladder(NDIM, ntemps=ntemps)
    out = np.tile(base, (NLEAVES, 1))
    out[:, 1:] *= 1.0 + 0.01 * rng.uniform(-1, 1, size=(NLEAVES, ntemps - 1))
    return out


def _build(info, sub):
    curr = SimpleNamespace(
        source_info={"emri": info},
        general_info=SimpleNamespace(ntemps=1, main_file_path=None),
    )
    state = SimpleNamespace(sub_states={"emri": sub})
    _, moves = _builder().build(None, curr, SimpleNamespace(acs_total_entries=NW), {}, state)
    return moves[0], state.sub_states["emri"]


class ResumeLadderTest(unittest.TestCase):
    LOGGER = "lisatools.globalfit.recipe"

    def test_fresh_store_tiles_the_configured_ladder(self):
        cfg = make_ladder(NDIM, ntemps=8)
        move, sub = _build(_info(cfg), _sub(8))
        np.testing.assert_array_equal(sub.betas_all, np.tile(cfg, (NLEAVES, 1)))
        np.testing.assert_array_equal(move.kwargs["betas_all"], sub.betas_all)
        self.assertEqual(move.coords_shape, (8, NW, NLEAVES, NDIM))

    def test_a_resume_keeps_the_stored_per_leaf_ladders_by_default(self):
        """The defect: the build re-tiled the configuration over the store."""
        cfg = make_ladder(NDIM, ntemps=8)
        stored = _adapted(8)
        with self.assertLogs(self.LOGGER, level="INFO") as cm:
            move, sub = _build(_info(cfg), _sub(8, stored))
        np.testing.assert_array_equal(sub.betas_all, stored)
        np.testing.assert_array_equal(move.kwargs["betas_all"], stored)
        self.assertTrue(any("keeps the STORED" in l for l in cm.output), cm.output)
        self.assertTrue(any("EMRI_LADDER_RESET_ON_RESUME=1" in l for l in cm.output))

    def test_the_reset_knob_replaces_every_leaf_and_logs_old_and_new(self):
        cfg = make_ladder(NDIM, ntemps=8)
        stored = _adapted(8)
        with self.assertLogs(self.LOGGER, level="INFO") as cm:
            move, sub = _build(_info(cfg, reset=True), _sub(8, stored))
        np.testing.assert_array_equal(sub.betas_all, np.tile(cfg, (NLEAVES, 1)))
        np.testing.assert_array_equal(move.kwargs["betas_all"], sub.betas_all)
        text = "\n".join(cm.output)
        self.assertIn("REPLACING the stored per-leaf ladders of 3 leaves", text)
        for leaf in range(NLEAVES):
            self.assertIn(f"[EMRI_LADDER] leaf {leaf} ladder reset: old ", text)
        self.assertIn(np.array2string(stored[1], precision=5), text)

    def test_the_9mo_case_old_geometric_store_to_make_ladder(self):
        """A 1/1.2^k 8-rung store resumed with RESET=1 lands on make_ladder."""
        old = np.tile(1 / 1.2 ** np.arange(8), (NLEAVES, 1))
        cfg = make_ladder(NDIM, ntemps=8)
        _, sub = _build(_info(cfg, reset=True), _sub(8, old))
        np.testing.assert_array_equal(sub.betas_all[2], cfg)
        _, sub = _build(_info(cfg), _sub(8, old))
        np.testing.assert_array_equal(sub.betas_all, old)

    def test_a_rung_count_mismatch_keeps_the_store_count(self):
        cfg = make_ladder(NDIM, ntemps=8)
        stored = _adapted(6)
        info = _info(cfg)
        with self.assertLogs(self.LOGGER, level="INFO"):
            move, sub = _build(info, _sub(6, stored))
        self.assertEqual(move.coords_shape[0], 6)
        np.testing.assert_array_equal(sub.betas_all, stored)   # every leaf, not leaf 0 tiled
        # reset under a mismatch, configured LONGER than the store: truncated to
        # the store's count (the builder's own truncation rule; for a
        # make_ladder ladder that is make_ladder at the store's count)
        info = _info(cfg, reset=True)
        with self.assertLogs(self.LOGGER, level="INFO"):
            move, sub = _build(info, _sub(6, stored))
        self.assertEqual(sub.betas_all.shape, (NLEAVES, 6))
        np.testing.assert_array_equal(sub.betas_all[0], cfg[:6])
        np.testing.assert_allclose(sub.betas_all[0], make_ladder(NDIM, ntemps=6), rtol=1e-14)
        self.assertEqual(len(info.betas), 6)
        # configured SHORTER than the store: make_ladder at the store's count + warning
        info = _info(make_ladder(NDIM, ntemps=4), reset=True)
        with self.assertLogs(self.LOGGER, level="WARNING") as cm:
            move, sub = _build(info, _sub(6, stored))
        self.assertEqual(move.coords_shape[0], 6)
        np.testing.assert_array_equal(sub.betas_all[1], make_ladder(NDIM, ntemps=6))
        self.assertTrue(any("does not fit the store's 6 rungs" in l for l in cm.output))

    def test_an_invalid_stored_ladder_is_not_kept(self):
        cfg = make_ladder(NDIM, ntemps=4)
        for bad in (np.zeros((NLEAVES, 4)), np.full((NLEAVES, 4), np.nan),
                    np.tile([1.0, 0.5, 0.6, 0.1], (NLEAVES, 1))):
            with self.subTest(bad=bad[0].tolist()):
                _, sub = _build(_info(cfg), _sub(4, bad))
                np.testing.assert_array_equal(sub.betas_all, np.tile(cfg, (NLEAVES, 1)))

    def test_the_reanchor_threshold_reaches_the_move(self):
        cfg = make_ladder(NDIM, ntemps=4)
        move, _ = _build(_info(cfg, reanchor=75.0), _sub(4))
        self.assertEqual(move.kwargs["ladder_reanchor_nats"], 75.0)


if __name__ == "__main__":
    unittest.main()
