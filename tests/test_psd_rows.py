"""PSDMove in one-walker replica mode: rows scatter, begin/publish replay on every rank."""

import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.fanout import ComputeService, WalkerFanout
from lisatools.globalfit.communication.ranks import RankRole, build_layout
from lisatools.globalfit.communication.rowfanout import RowFanout
from lisatools.globalfit.moves.psdmove import PSDMove


class _Container:
    def __init__(self):
        self.sens_mat = "old"


class _Acs:
    def __init__(self):
        self.c = [_Container()]
        self.resets = 0

    def __getitem__(self, i):
        return self.c[i]

    def __len__(self):
        return 1

    def reset_linear_psd_arr(self):
        self.resets += 1


def _stub(tag):
    m = PSDMove.__new__(PSDMove)
    m.sampled_branches = ["galfor"]
    m.gf_move_name = "galfor_pe"
    m.likelihood_fanout = True
    m.row_fanout = None
    m.permute_every = 1
    m.acs = _Acs()
    m.coarse_runtime = None
    m._fixed_noise_coords = {}
    m.log = []
    m._score_rows = lambda w, p, g, s, _t=tag: (
        np.asarray(p).sum(axis=1) + (0.0 if g is None else np.asarray(g).sum(axis=1)) + _t
    )
    m._prepare_fixed_component_covariances = lambda: m.log.append(("prep", dict(m._fixed_noise_coords)))
    m._build_sensitivity_for_walker = lambda w, p, g, s: ("sens", w, tuple(np.asarray(p)), g, s)
    return m


def _world(head_fn, knob=None):
    world = FakeWorld(3, nodes=[0, 0, 0])
    moves = {}

    def fn(rank, comm):
        layout = build_layout(comm, 1, [0, 1], legacy=False)
        fcomm = layout.make_fanout_comm(comm)
        if layout.role_of(rank) == RankRole.SAVER:
            return "saver"
        fo = WalkerFanout(fcomm, layout, rank, model=None)
        move = _stub(100.0 * rank)
        move.row_fanout = RowFanout(fo, move)
        if knob is not None:
            move.likelihood_fanout = knob
        moves[rank] = move
        if layout.role_of(rank) == RankRole.HEAD:
            fo.enter_stage("pe", "pe")
            try:
                return head_fn(move)
            finally:
                fo.stop()
        return ComputeService(fcomm, layout, rank, registry={("pe", "galfor_pe"): move}).serve()

    return world.run(fn), moves


PSD = np.arange(10.0).reshape(5, 2)
GAL = np.ones((5, 3))
W = np.zeros(5, dtype=int)


class PSDRowsTest(unittest.TestCase):
    def test_rows_scatter_and_absent_branches_stay_none(self):
        out, _ = _world(lambda m: (m.compute_psd_rows(W, PSD, GAL, None), m.compute_psd_rows(W, PSD, None, None)))
        with_gal, without = out[0]
        tags = np.array([0, 0, 0, 100, 100], dtype=float)
        np.testing.assert_array_equal(with_gal, PSD.sum(axis=1) + 3.0 + tags)
        np.testing.assert_array_equal(without, PSD.sum(axis=1) + tags)

    def test_begin_and_publish_replay_on_every_rank(self):
        fixed = {"psd": np.array([[0.5, 0.25]])}

        def head(m):
            m._fixed_noise_coords = fixed
            m._replay_noise_begin()
            state = SimpleNamespace(branches_coords={
                "psd": np.array([[[[1.0, 2.0]]]]), "galfor": np.array([[[[3.0, 4.0, 5.0]]]]),
            })
            m._replay_noise_publish(state)
            return m.log, m.acs[0].sens_mat, m.acs.resets

        out, moves = _world(head)
        log, sens, resets = out[0]
        self.assertEqual(log[0][0], "prep")
        np.testing.assert_array_equal(log[0][1]["psd"], fixed["psd"])
        self.assertEqual(sens[:3], ("sens", 0, (1.0, 2.0)))
        np.testing.assert_array_equal(sens[3], [3.0, 4.0, 5.0])
        self.assertIsNone(sens[4])
        self.assertEqual(resets, 1)
        # the replica applied the same two steps
        self.assertEqual(moves[1].log[0][0], "prep")
        np.testing.assert_array_equal(moves[1]._fixed_noise_coords["psd"], fixed["psd"])
        self.assertEqual(moves[1].acs[0].sens_mat[:3], ("sens", 0, (1.0, 2.0)))
        self.assertEqual(moves[1].acs.resets, 1)

    def test_knob_off_still_replays_begin_and_publish(self):
        """F1: {PREFIX}_LIKELIHOOD_FANOUT=0 must not stop begin/publish replays.

        The knob only routes SCORING to the head; the fixed-noise-coords
        prep and the sens_mat publish must still reach every replica.
        """
        fixed = {"psd": np.array([[0.5, 0.25]])}

        def head(m):
            m._fixed_noise_coords = fixed
            m._replay_noise_begin()
            state = SimpleNamespace(branches_coords={
                "psd": np.array([[[[1.0, 2.0]]]]), "galfor": np.array([[[[3.0, 4.0, 5.0]]]]),
            })
            m._replay_noise_publish(state)
            return m.log, m.acs[0].sens_mat, m.acs.resets

        out, moves = _world(head, knob=False)
        log, sens, resets = out[0]
        self.assertEqual(log[0][0], "prep")
        self.assertEqual(sens[:3], ("sens", 0, (1.0, 2.0)))
        self.assertEqual(resets, 1)
        # the replica applied the same two steps even with the knob off
        self.assertEqual(moves[1].log[0][0], "prep")
        np.testing.assert_array_equal(moves[1]._fixed_noise_coords["psd"], fixed["psd"])
        self.assertEqual(moves[1].acs[0].sens_mat[:3], ("sens", 0, (1.0, 2.0)))
        self.assertEqual(moves[1].acs.resets, 1)

    def test_knob_prefix_and_fancy_gate(self):
        m = _stub(0.0)
        self.assertEqual(m.fanout_knob_prefix(), "GALFOR")
        self.assertFalse(m._fancy_swap_fires(0, 1))
        self.assertTrue(m._fancy_swap_fires(0, 4))
        self.assertTrue(m._fancy_swap_fires(1, 4))  # permute_every=1 -> every move_i
        m.permute_every = 2
        self.assertFalse(m._fancy_swap_fires(1, 4))
        self.assertTrue(m._fancy_swap_fires(2, 4))
        for off in (-1, 0):  # explicitly off; -1 used to fire every move_i, 0 divided by zero
            m.permute_every = off
            self.assertFalse(any(m._fancy_swap_fires(i, 4) for i in range(6)))

    def test_single_process_paths_are_direct(self):
        m = _stub(1.0)
        np.testing.assert_array_equal(m.compute_psd_rows(W, PSD, None, None), PSD.sum(axis=1) + 1.0)
        m._fixed_noise_coords = {"psd": np.zeros((1, 2))}
        m._replay_noise_begin()
        self.assertEqual(m.log[0][0], "prep")

    def test_kernel_tier_supps_is_a_plain_dict(self):
        """Regression: the real ``_score_rows`` kernel tier, unstubbed.

        ``BranchSupplemental`` is not string-indexable; ``psd_log_like``
        indexes ``supps["walker_inds"]`` directly, so the seam must hand it a
        plain dict (matching the old ``supps[logp_keep]`` slice).
        """
        m = PSDMove.__new__(PSDMove)
        m._kernel_fast_path_available = lambda has_sgwb=False: True
        m.psd_kwargs = {}
        seen = {}

        def fake_psd_log_like(input_args, supps=None, **kw):
            seen["supps"] = supps
            return input_args[0].sum(axis=1)

        m.psd_log_like = fake_psd_log_like
        out = m._score_rows(W, PSD, None, None)
        np.testing.assert_array_equal(seen["supps"]["walker_inds"], W)
        np.testing.assert_array_equal(out, PSD.sum(axis=1))


if __name__ == "__main__":
    unittest.main()
