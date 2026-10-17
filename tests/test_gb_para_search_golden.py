"""With GB_PARA_SEARCH off the PE draw-one is BIT-IDENTICAL to HEAD 57df9f6e.

The parallel search touches the shared PE RJ draw (``GFCombineMove.
_pe_rj_draw_one_config`` / ``_pe_rj_draw_one_plan``: a fifth member,
``rj_para_search_pe``, appended LAST with its own share
``GB_PE_RJ_PARA_FRACTION``, default 0). ``tests/data/
para_search_draw_one_golden.npz`` was written by :func:`write_golden` against
a ``git archive 57df9f6e`` export of ``src/`` (the pre-change code, put
first on the import path ahead of the editable finder), and the current code
must reproduce every executed-move sequence exactly:

* the production compositions (warm + one or two F-stat moves + prior +
  ridge) at the production fractions;
* a stage that CARRIES ``rj_para_search_pe`` with the share unset (HEAD
  treats the unknown name as an ordinary wrapped move that runs every
  iteration -- exactly the share-0 semantics) and with
  ``GB_PE_RJ_PARA_FRACTION=0`` set explicitly;
* the draw-one switched off (the weighted cycle).

The other off-path changes (the ``"para_search": None`` payload key, the
no-op ``_para_search_apply`` / ``_para_search_note`` hooks of every GB move)
consume no random numbers and change no state; the recipe builds no move and
composes nothing without ``gb.para_search``.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                      "para_search_draw_one_golden.npz")
N_PROPOSES = 400

WARM, F1, F2, PRIOR, RIDGE, PARA = ("rj_warm_pe", "rj_fstat_pe",
                                    "rj_fstat_pe_residual", "rj_prior_pe",
                                    "gb_ridge_gibbs", "rj_para_search_pe")
CODES = {n: i for i, n in enumerate((WARM, F1, F2, PRIOR, RIDGE, PARA))}

#: name -> (stage move names, env, seed)
GOLDEN_SCENARIOS = {
    "prod_two_fstat": ((WARM, F1, F2, PRIOR, RIDGE),
                       dict(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_FRACTION=0.45,
                            GB_PE_RJ_WARM_FRACTION=0.1,
                            GB_PE_RJ_FSTAT_RESIDUAL_SPLIT=0.5), 11),
    "prod_one_fstat": ((WARM, F1, PRIOR, RIDGE),
                       dict(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_FRACTION=0.8), 12),
    "with_para_unset": ((WARM, F1, F2, PRIOR, PARA, RIDGE),
                        dict(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_FRACTION=0.45,
                             GB_PE_RJ_WARM_FRACTION=0.1), 13),
    "with_para_zero": ((WARM, F1, F2, PRIOR, PARA, RIDGE),
                       dict(GB_PE_RJ_DRAW_ONE=1, GB_PE_RJ_FSTAT_FRACTION=0.45,
                            GB_PE_RJ_WARM_FRACTION=0.1, GB_PE_RJ_PARA_FRACTION=0), 14),
    "draw_one_off": ((WARM, F1, F2, PRIOR, PARA, RIDGE),
                     dict(GB_PE_RJ_DRAW_ONE=0), 15),
}

_KNOBS = ("GB_PE_RJ_DRAW_ONE", "GB_PE_RJ_FSTAT_FRACTION", "GB_PE_RJ_WARM_FRACTION",
          "GB_PE_RJ_FSTAT_RESIDUAL_SPLIT", "GB_PE_RJ_PARA_FRACTION", "GF_MOVE_TIMING")


class _StubMove:
    temperature_control = None
    periodic = None

    def __init__(self, name, log):
        self.gf_move_name = name
        self.log = log
        self.accepted = np.zeros((2, 4))

    def propose(self, model, state):
        self.log.append(CODES[self.gf_move_name])
        return state, np.ones((2, 4))


class _StubState:
    sub_states = None


def golden_stream(name, n=N_PROPOSES):
    """``(n, 6)`` int array: the executed move codes per propose (-1 padded)."""
    from lisatools.globalfit.moves.globalfitmove import GFCombineMove

    names, env, seed = GOLDEN_SCENARIOS[name]
    clean = {k: v for k, v in os.environ.items() if k not in _KNOBS}
    clean.update({k: str(v) for k, v in env.items()})
    log = []
    with mock.patch.dict(os.environ, clean, clear=True):
        comb = GFCombineMove(moves=[_StubMove(m, log) for m in names],
                             share_temperature_control=False, weighted_cycle=True)
        comb.gf_stage_kind = "pe"
        comb.gf_stage_name = "full_pe"
        model = SimpleNamespace(random=np.random.RandomState(seed))
        out = np.full((n, 6), -1, dtype=np.int64)
        for i in range(n):
            log.clear()
            comb.propose(model, _StubState())
            out[i, :len(log)] = log
    return out


def write_golden(path=GOLDEN):
    """Regenerate the golden file (run against the PRE-change code only)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{name: golden_stream(name) for name in GOLDEN_SCENARIOS})
    return path


class DrawOneGoldenTest(unittest.TestCase):
    def test_off_paths_are_bit_identical_to_head(self):
        ref = np.load(GOLDEN)
        self.assertEqual(sorted(ref.files), sorted(GOLDEN_SCENARIOS))
        for name in GOLDEN_SCENARIOS:
            with self.subTest(scenario=name):
                np.testing.assert_array_equal(golden_stream(name), ref[name])

    def test_a_positive_share_changes_the_stream(self):
        """The negative control: the same stage at a positive share must NOT
        reproduce the golden (else the comparison above would be vacuous)."""
        names, env, seed = GOLDEN_SCENARIOS["with_para_unset"]
        GOLDEN_SCENARIOS["_ctl"] = (names, dict(env, GB_PE_RJ_PARA_FRACTION=0.2), seed)
        try:
            got = golden_stream("_ctl")
        finally:
            GOLDEN_SCENARIOS.pop("_ctl")
        ref = np.load(GOLDEN)["with_para_unset"]
        self.assertFalse(np.array_equal(got, ref))


if __name__ == "__main__":
    unittest.main()
