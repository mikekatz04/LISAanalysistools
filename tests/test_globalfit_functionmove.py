"""Unit tests for FunctionMove (plain-function global-fit moves).

Everything here runs on stubs — no build, no data, no sampler.
"""

import copy
import pickle
import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit import FunctionMove, MoveBuildContext

NTEMPS, NWALKERS = 2, 4


class _StubACS:
    """Stands in for an AnalysisContainerArray: fixed per-walker likelihood."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def likelihood(self, complex=False):
        return self.values


def _state():
    return SimpleNamespace(log_like=np.zeros((NTEMPS, NWALKERS)))


def _model(acs):
    return SimpleNamespace(analysis_container_arr=acs, map_fn=map, random=np.random)


def _ctx(acs=None):
    return MoveBuildContext(
        recipe=None, engine_info=None, curr=None, acs=acs, priors={}, state=None,
        stock_moves={}, ntemps=NTEMPS, nwalkers=NWALKERS,
    )


def module_level_move(model, state):
    return state, None


class FunctionMoveTest(unittest.TestCase):
    def test_requires_callable(self):
        with self.assertRaises(TypeError):
            FunctionMove("not a function")

    def test_name_defaults_to_fn_name(self):
        self.assertEqual(FunctionMove(module_level_move).name, "module_level_move")
        self.assertEqual(FunctionMove(module_level_move, name="mine").name, "mine")

    def test_setup_captures_run_objects(self):
        acs = _StubACS(np.arange(NWALKERS))
        fm = FunctionMove(module_level_move)
        self.assertIs(fm.materialize(_ctx(acs)), fm)  # setup returns None -> self
        self.assertIs(fm.acs, acs)
        self.assertEqual(fm.accepted.shape, (NTEMPS, NWALKERS))

    def test_output_normalization(self):
        acs = _StubACS(np.zeros(NWALKERS))
        model = _model(acs)

        for fn, desc in [
            (lambda m, s: (s, None), "tuple with None accepted"),
            (lambda m, s: s, "bare state"),
            (lambda m, s: None, "None -> keep input state"),
        ]:
            with self.subTest(desc):
                fm = FunctionMove(fn, name="t")
                state = _state()
                new_state, accepted = fm.propose(model, state)
                self.assertIs(new_state, state)
                np.testing.assert_array_equal(accepted, np.zeros((NTEMPS, NWALKERS)))

        ones = np.ones((NTEMPS, NWALKERS))
        fm = FunctionMove(lambda m, s: (s, ones), name="t")
        _, accepted = fm.propose(model, _state())
        np.testing.assert_array_equal(accepted, ones)

    def test_log_like_sync(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        model = _model(_StubACS(values))
        fm = FunctionMove(module_level_move)
        state = _state()
        new_state, _ = fm.propose(model, state)
        # broadcast per-walker likelihood over all temperature rungs
        np.testing.assert_array_equal(
            new_state.log_like, np.tile(values, (NTEMPS, 1))
        )

    def test_log_like_sync_opt_out(self):
        model = _model(_StubACS(np.full(NWALKERS, 7.0)))
        fm = FunctionMove(module_level_move, sync_log_like=False)
        state = _state()
        new_state, _ = fm.propose(model, state)
        np.testing.assert_array_equal(new_state.log_like, np.zeros((NTEMPS, NWALKERS)))

    def test_sync_falls_back_to_setup_acs(self):
        values = np.arange(NWALKERS, dtype=float)
        fm = FunctionMove(module_level_move)
        fm.materialize(_ctx(_StubACS(values)))
        model = SimpleNamespace(analysis_container_arr=None)  # model carries no acs
        new_state, _ = fm.propose(model, _state())
        np.testing.assert_array_equal(new_state.log_like, np.tile(values, (NTEMPS, 1)))

    def test_sync_without_any_acs_errors(self):
        fm = FunctionMove(module_level_move)
        with self.assertRaises(RuntimeError):
            fm.propose(SimpleNamespace(analysis_container_arr=None), _state())

    def test_pickle_drops_runtime_objects(self):
        fm = FunctionMove(module_level_move, branch="line")
        fm.materialize(_ctx(_StubACS(np.zeros(NWALKERS))))
        clone = pickle.loads(pickle.dumps(copy.deepcopy(fm)))
        self.assertEqual(clone.name, "module_level_move")
        self.assertEqual(clone.branch, "line")
        self.assertIsNone(clone.acs)
        self.assertIsNone(clone.runtime)


if __name__ == "__main__":
    unittest.main()
