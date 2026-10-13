"""The lost-rung guard inside REAL proposes, single process and fanned out.

``tests/test_addremove_ladder_guard.py`` pins ``_reanchor_lost_rungs`` on a
fake. Here the REAL ``EMRIDirectLikeMove`` of ``tests/test_emri_direct_fanout``
(tiny WDM grid, two leaves, per-walker different noise and PSD, 2 rungs at
betas [1, 0.5]) starts with ONE lost hot rung -- leaf 1, rung 1 of one walker
moved to a merger 28 layers away from its source -- and runs two proposes:

* single process: the guard fires at that leaf's first visit, names the
  walker / rung, and the rung is no longer lost at the end of the propose; the
  paired control (``EMRI_LADDER_REANCHOR_NATS=0``) leaves it lost;
* WALKER BLOCKS (4 walkers on 4 ranks, the production layout): the rank that
  owns the walker fires with the GLOBAL walker index, and the head's merged
  state equals the single-process run bit for bit (the edit is a
  within-walker, within-leaf one, so ``merge_walkers`` carries it);
* ONE-WALKER REPLICAS (1 walker on 2 ranks): the head's flat body fires;
  states equal the single-process run.

The threshold is set between the probe-measured deficit of the lost rung and
the largest deficit of every other rung, so only the planted rung can fire.
"""

from __future__ import annotations

import os
import threading
import unittest
from unittest import mock

import numpy as np

try:
    from tests import test_emri_direct_fanout as E
except ImportError:  # pragma: no cover - run from tests/
    import test_emri_direct_fanout as E

from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove

H = E.H
LEAF = 1
RUNG = 1
#: expected tempering drop of rung 1 (beta 0.5, ndim 11): 5.5 * (2 - 1)
MARGIN = 0.5 * H.NDIM * (1.0 / H.BETAS[1] - 1.0)


def _far(w, leaf):
    row = H._hot(w, leaf).copy()
    row[10] = (H.LEAF_LAYERS[leaf] - 28.0 + 0.47) * H.LAYER   # still inside the prior box
    return row


def _planted_state(lost_walker):
    real = E._state

    def _state(walkers):
        state = real(walkers)
        walkers = list(walkers)
        if lost_walker in walkers:
            j = walkers.index(lost_walker)
            state.sub_states["emri"].coords[RUNG, j, LEAF] = _far(lost_walker, LEAF)
        return state

    return _state


class _Recorder:
    """Class-level wrapper of the guard: every call on every FakeWorld thread."""

    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()
        self._real = ResidualAddOneRemoveOneMove._reanchor_lost_rungs

    def __call__(self, move, work, leaf, prev_logl, prev_logp, betas):
        ll = np.array(prev_logl, dtype=float, copy=True)
        out = self._real(move, work, leaf, prev_logl, prev_logp, betas)
        with self._lock:
            self.calls.append(dict(leaf=int(leaf), deficit=ll[0][None, :] - ll[1:],
                                   records=list(out[2]), w0=move._walker_offset()))
        return out

    def patch(self):
        rec = self

        def wrapper(move, *a):
            return rec(move, *a)

        return mock.patch.object(ResidualAddOneRemoveOneMove, "_reanchor_lost_rungs", wrapper)

    def fired(self):
        return sorted((c["leaf"],) + r[:2] for c in self.calls for r in c["records"])


class _GuardCase(E._Case):
    LOST_WALKER = 2
    NWALKERS = 4

    def _kw(self):
        raise NotImplementedError

    def _run_single(self, thresh, rec=None):
        rec = rec or _Recorder()
        with mock.patch.dict(os.environ, {"EMRI_LADDER_REANCHOR_NATS": str(thresh)}), \
                mock.patch.object(E, "_state", _planted_state(self.LOST_WALKER)), rec.patch(), \
                self._ctx():
            out = self._single(self.NWALKERS, **self._kw())
        return out, rec

    def _ctx(self):
        return mock.patch.dict({})          # no-op context

    def _threshold(self):
        """Probe with the guard effectively off; split lost from everything else."""
        _, probe = self._run_single(1e300)
        lost, other = [], []
        for c in probe.calls:
            d = c["deficit"]
            for (k, w), val in np.ndenumerate(d):
                if c["leaf"] == LEAF and k + 1 == RUNG and w + c["w0"] == self.LOST_WALKER:
                    lost.append(val)
                else:
                    other.append(val)
        self.assertEqual(len(lost), self.N_PROPOSES)        # still lost at every visit
        d_lost, d_other = min(lost), max(other)
        self.assertGreater(d_lost, d_other + 4 * MARGIN, (d_lost, d_other))
        return 0.5 * (d_lost + d_other) - MARGIN, d_lost

    def _lost_deficit(self, state):
        sub = state.sub_states["emri"]
        return float(sub.log_like[LEAF, 0, self.LOST_WALKER] - sub.log_like[LEAF, RUNG, self.LOST_WALKER])

    def _check(self, world_ranks):
        thresh, d_lost = self._threshold()
        # single process, guard on: fires once, at the planted rung
        ref, rec = self._run_single(thresh)
        self.assertEqual(rec.fired(), [(LEAF, self.LOST_WALKER, RUNG)])
        self.assertLess(self._lost_deficit(ref["states"][0]), thresh)
        # paired control: guard off -> the rung is still lost after both proposes
        ctrl, rec0 = self._run_single(0)
        self.assertEqual(rec0.fired(), [])
        self.assertGreater(self._lost_deficit(ctrl["states"][-1]), thresh + MARGIN)
        # fanned out: same firing (GLOBAL walker index), same merged states
        rec_w = _Recorder()
        with mock.patch.dict(os.environ, {"EMRI_LADDER_REANCHOR_NATS": str(thresh)}), \
                mock.patch.object(E, "_state", _planted_state(self.LOST_WALKER)), rec_w.patch(), \
                self._ctx():
            out = self._world(world_ranks, self.NWALKERS, **self._kw())
        self.assertEqual(rec_w.fired(), [(LEAF, self.LOST_WALKER, RUNG)])
        self._assert_match(out, ref)
        head = out[0]["states"]
        self.assertLess(self._lost_deficit(head[0]), thresh)
        return out, rec_w


class WalkerBlockGuardTest(_GuardCase):
    LOST_WALKER = 2
    NWALKERS = 4

    def _kw(self):
        return dict(scope="per_walker", adaptive=False, refresh=10, rng=H._RowLocalRandom,
                    permute_every=0)

    def _ctx(self):
        return H._row_local_numpy()

    def test_the_owning_rank_fires_and_the_head_merges_it(self):
        out, rec = self._check(4)
        layout = out[0]["layout"]
        self.assertFalse(layout.replica_mode)
        fired = [c for c in rec.calls if c["records"]]
        self.assertEqual(len(fired), 1)
        # the rank whose block holds walker 2 fired, from its LOCAL walker 0
        self.assertEqual(fired[0]["w0"], 2)


class ReplicaGuardTest(_GuardCase):
    LOST_WALKER = 0
    NWALKERS = 1

    def _kw(self):
        return dict(scope="walker_max", permute_every=1, adaptive=True,
                    rng=lambda: np.random.RandomState(H.SEED))

    def test_the_flat_body_fires_on_the_head(self):
        out, rec = self._check(2)
        self.assertTrue(out[0]["layout"].replica_mode)


if __name__ == "__main__":
    unittest.main()
