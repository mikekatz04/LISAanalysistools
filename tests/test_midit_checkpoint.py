"""Mid-iteration checkpointing: the preemption-protection sidecar.

Hermetic and CPU-only. The module treats the state as an opaque
picklable object, so these tests drive it with plain containers; the
``run.py`` config-compatibility gate is exercised with a fake ``self``
(no engine build). The scenario the feature exists for is pinned in the
docstrings: 2026-08-27, five spot preemptions in one day against an
~85-minute exposure to the first ``[SAVE]`` -- zero iterations stored.
"""

import os
import pickle
import tempfile
import types
import unittest

import numpy as np

from lisatools.globalfit import midit_checkpoint as mc


class _Cleanly(unittest.TestCase):
    """Every test runs disarmed-in, disarmed-out, in its own tmpdir."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.main = os.path.join(self._tmp.name, "run_main.h5")
        mc.disarm()

    def tearDown(self):
        mc.disarm()
        self._tmp.cleanup()


class PathAndArmTest(_Cleanly):
    def test_checkpoint_path_is_a_pkl_sidecar(self):
        self.assertEqual(
            mc.checkpoint_path("/a/b/run_main.h5"),
            "/a/b/run_main_midit_checkpoint.pkl",
        )

    def test_unarmed_maybe_write_is_a_noop(self):
        self.assertFalse(mc.maybe_write({"x": 1}, tag="t"))
        self.assertFalse(os.path.exists(mc.checkpoint_path(self.main)))

    def test_armed_flag(self):
        self.assertFalse(mc.armed())
        mc.arm(self.main)
        self.assertTrue(mc.armed())
        mc.disarm()
        self.assertFalse(mc.armed())


class WriteLoadRoundtripTest(_Cleanly):
    def test_roundtrip(self):
        mc.arm(self.main, min_interval=0.0, stored_iteration=3)
        state = {"coords": np.arange(12.0).reshape(3, 4)}
        self.assertTrue(mc.maybe_write(state, tag="mbh leaf 2"))
        got = mc.load_for_resume(self.main, stored_iteration=3)
        self.assertIsNotNone(got)
        got_state, meta = got
        np.testing.assert_array_equal(got_state["coords"], state["coords"])
        self.assertEqual(meta["stored_iteration"], 3)
        self.assertEqual(meta["tag"], "mbh leaf 2")

    def test_missing_file_returns_none(self):
        self.assertIsNone(mc.load_for_resume(self.main, 0))

    def test_throttle_skips_and_force_overrides(self):
        mc.arm(self.main, min_interval=3600.0)
        # inside the interval (clock starts at arm): skipped
        self.assertFalse(mc.maybe_write({"a": 1}, tag="early"))
        self.assertFalse(os.path.exists(mc.checkpoint_path(self.main)))
        # force writes regardless
        self.assertTrue(mc.maybe_write({"a": 2}, tag="forced", force=True))
        _, meta = mc.load_for_resume(self.main, 0)
        self.assertEqual(meta["tag"], "forced")

    def test_prepare_runs_only_when_a_write_is_due(self):
        """prepare is the expensive cold-row sync -- never pay it inside
        the throttle window."""
        calls = []
        mc.arm(self.main, min_interval=3600.0)
        mc.maybe_write({"a": 1}, prepare=lambda st: calls.append("skip"))
        self.assertEqual(calls, [])
        mc.maybe_write({"a": 1}, prepare=lambda st: calls.append("due"),
                       force=True)
        self.assertEqual(calls, ["due"])

    def test_failed_pickle_never_raises_and_leaves_previous_intact(self):
        mc.arm(self.main, min_interval=0.0)
        self.assertTrue(mc.maybe_write({"good": 1}, tag="first"))
        # a lambda is unpicklable -> the write fails, silently to the caller
        self.assertFalse(mc.maybe_write({"bad": lambda: None}, tag="second"))
        got_state, meta = mc.load_for_resume(self.main, 0)
        self.assertEqual(got_state, {"good": 1})
        self.assertEqual(meta["tag"], "first")
        # no torn tmp left behind
        self.assertFalse(
            os.path.exists(mc.checkpoint_path(self.main) + ".tmp"))


class PrecedenceTest(_Cleanly):
    """The checkpoint wins iff written at (or after) the store's newest
    stored iteration; note_saved() is how the store keeps score."""

    def test_stale_checkpoint_is_ignored(self):
        mc.arm(self.main, min_interval=0.0, stored_iteration=5)
        mc.maybe_write({"s": 1})
        # store has moved on to 6: the checkpoint's content is already
        # contained in the store
        self.assertIsNone(mc.load_for_resume(self.main, stored_iteration=6))
        # ...but NOT moved aside -- staleness is not corruption
        self.assertTrue(os.path.exists(mc.checkpoint_path(self.main)))

    def test_equal_iteration_checkpoint_wins(self):
        """Killed mid-iteration N+1 with N stored: exactly the probe's
        life story -- the checkpoint carries the partial progress."""
        mc.arm(self.main, min_interval=0.0, stored_iteration=5)
        mc.maybe_write({"s": 1})
        self.assertIsNotNone(mc.load_for_resume(self.main, stored_iteration=5))

    def test_note_saved_ticks_the_stored_count(self):
        mc.arm(self.main, min_interval=0.0, stored_iteration=0)
        mc.note_saved()
        mc.note_saved()
        mc.maybe_write({"s": 1})
        _, meta = mc.load_for_resume(self.main, stored_iteration=2)
        self.assertEqual(meta["stored_iteration"], 2)

    def test_note_saved_unarmed_is_a_noop(self):
        mc.note_saved()  # must not raise


class RejectionTest(_Cleanly):
    def _ckpt(self):
        return mc.checkpoint_path(self.main)

    def test_corrupt_file_moved_aside(self):
        with open(self._ckpt(), "wb") as fh:
            fh.write(b"not a pickle")
        self.assertIsNone(mc.load_for_resume(self.main, 0))
        self.assertFalse(os.path.exists(self._ckpt()))
        self.assertTrue(os.path.exists(self._ckpt() + ".rejected"))

    def test_wrong_format_version_moved_aside(self):
        with open(self._ckpt(), "wb") as fh:
            pickle.dump({"meta": {"format": -999, "stored_iteration": 0},
                         "state": {}}, fh)
        self.assertIsNone(mc.load_for_resume(self.main, 0))
        self.assertTrue(os.path.exists(self._ckpt() + ".rejected"))

    def test_validate_rejection_moved_aside(self):
        mc.arm(self.main, min_interval=0.0)
        mc.maybe_write({"s": 1})
        self.assertIsNone(mc.load_for_resume(
            self.main, 0, validate=lambda st: (False, "6-rung ladder vs 4")))
        self.assertTrue(os.path.exists(self._ckpt() + ".rejected"))

    def test_validate_pass_returns_state(self):
        mc.arm(self.main, min_interval=0.0)
        mc.maybe_write({"s": 1})
        got = mc.load_for_resume(self.main, 0, validate=lambda st: (True, ""))
        self.assertIsNotNone(got)


class SelfTestTest(_Cleanly):
    """The startup self-test: a run proves the checkpoint path works in
    situ (GPU node / MPI layout / branch set the unit tests cannot reach)
    before it has anything to lose."""

    def test_passes_on_a_picklable_state(self):
        mc.arm(self.main, min_interval=3600.0)
        self.assertTrue(mc.self_test({"coords": np.zeros(4)}))
        self.assertTrue(os.path.exists(mc.checkpoint_path(self.main)))

    def test_ignores_the_throttle(self):
        """arm() starts the clock, so an un-forced write would be skipped;
        the self-test must still run."""
        mc.arm(self.main, min_interval=99999.0)
        self.assertTrue(mc.self_test({"a": 1}))

    def test_fails_on_unpicklable_state_without_raising(self):
        mc.arm(self.main, min_interval=0.0)
        self.assertFalse(mc.self_test({"bad": lambda: None}))

    def test_fails_when_the_validate_gate_rejects_our_own_state(self):
        """Gate rejecting the run's own state = a bug in the gate; the
        self-test is what surfaces it, at startup rather than at resume."""
        mc.arm(self.main, min_interval=0.0)
        self.assertFalse(
            mc.self_test({"a": 1}, validate=lambda st: (False, "bogus gate")))

    def test_unarmed_is_false(self):
        self.assertFalse(mc.self_test({"a": 1}))

    def test_does_not_disturb_the_stored_counter(self):
        mc.arm(self.main, min_interval=0.0, stored_iteration=7)
        self.assertTrue(mc.self_test({"a": 1}))
        _, meta = mc.load_for_resume(self.main, stored_iteration=7)
        self.assertEqual(meta["stored_iteration"], 7)


class RunValidateGateTest(unittest.TestCase):
    """The run.py config-compatibility gate, driven with a fake ``self``.

    Shapes mirror the 6-mo sources probe: engine cold chain (1, 24),
    mbh 4 leaves x 11 dims on a 4-rung branch ladder.
    """

    NW, NT_ENGINE, NT_BRANCH = 24, 1, 4

    def _fake_self(self):
        return types.SimpleNamespace(
            engine_info=types.SimpleNamespace(
                branch_names=["mbh"], nleaves_max={"mbh": 4}),
            curr=types.SimpleNamespace(
                ndims={"mbh": 11}, source_info={}),
            nwalkers=self.NW,
            ntemps=self.NT_ENGINE,
            _branch_ntemps=lambda name: self.NT_BRANCH,
        )

    def _fake_state(self, ndim=11, nleaves=4, nt_branch=4):
        return types.SimpleNamespace(
            branches={"mbh": types.SimpleNamespace(
                coords=np.zeros((self.NT_ENGINE, self.NW, nleaves, ndim)))},
            log_like=np.zeros((self.NT_ENGINE, self.NW)),
            sub_states={"mbh": types.SimpleNamespace(
                betas_all=np.zeros((nleaves, nt_branch)), band_info=None)},
        )

    def _validate(self, state):
        from lisatools.globalfit.run import GlobalFit

        return GlobalFit._midit_checkpoint_validate(self._fake_self(), state)

    def test_matching_config_passes(self):
        ok, why = self._validate(self._fake_state())
        self.assertTrue(ok, why)

    def test_ladder_change_rejected(self):
        """THE landmine: a 6-rung-era snapshot into a 4-rung relaunch."""
        ok, why = self._validate(self._fake_state(nt_branch=6))
        self.assertFalse(ok)
        self.assertIn("ladder", why)

    def test_ndim_change_rejected(self):
        ok, why = self._validate(self._fake_state(ndim=9))
        self.assertFalse(ok)
        self.assertIn("coords", why)

    def test_branch_set_change_rejected(self):
        state = self._fake_state()
        state.branches["gb"] = state.branches["mbh"]
        ok, why = self._validate(state)
        self.assertFalse(ok)
        self.assertIn("branch set", why)


if __name__ == "__main__":
    unittest.main()
