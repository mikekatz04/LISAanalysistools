"""Per-class in-model repeat budgets + de-synced accept chain (2026-08-15).

Lightweight fake-based tests (same style as ``test_rj_flip_fraction``):

* knob resolution for ``inmodel_repeats_{newborn,survivor}``
  (kwarg > env > builder default > mode default; search vs PE modes);
* pooled-survivor provenance partition (``_split_by_newborn`` + the
  pick-time newborn mask expression used by the direct-batch pooling);
* the de-synced in-model accept chain reproduces a straight-line
  reference (the pre-2026-08-15 host-gated logic) bit-for-bit on a tiny
  fake: same accept decisions, same tracked state, same counters.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import (
    GBSpecialStretchMove,
    _inmodel_repeats_mode_defaults,
    _picked_batches,
    _resolve_inmodel_repeats,
    _split_by_newborn,
)


# --------------------------------------------------------------------------
# knob resolution
# --------------------------------------------------------------------------
class InmodelRepeatKnobTest(unittest.TestCase):
    def test_kwarg_wins_over_env_and_default(self):
        with mock.patch.dict(os.environ, {"GB_INMODEL_REPEATS_NEWBORN": "7"}):
            self.assertEqual(
                _resolve_inmodel_repeats("gb", "newborn", 13, 200), 13
            )

    def test_env_wins_over_default(self):
        with mock.patch.dict(os.environ, {"GB_INMODEL_REPEATS_NEWBORN": "7"}):
            self.assertEqual(
                _resolve_inmodel_repeats("gb", "newborn", None, 200), 7
            )

    def test_default_fallback_and_branch_prefix(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_INMODEL_REPEATS_SURVIVOR", None)
            self.assertEqual(
                _resolve_inmodel_repeats("gb", "survivor", None, 25), 25
            )
        # a foreign branch's env must not leak in
        with mock.patch.dict(
            os.environ, {"GB_INMODEL_REPEATS_SURVIVOR": "9"}
        ):
            self.assertEqual(
                _resolve_inmodel_repeats("vgb", "survivor", None, 25), 25
            )

    def test_below_one_rejected(self):
        with self.assertRaises(ValueError):
            _resolve_inmodel_repeats("gb", "newborn", 0, 200)

    def test_mode_defaults_search(self):
        with mock.patch.dict(os.environ, {"GB_MODE": "search"}):
            self.assertEqual(
                _inmodel_repeats_mode_defaults("gb", 100), (200, 25)
            )

    def test_mode_defaults_pe_follows_num_repeat_proposals(self):
        for env in ({}, {"GB_MODE": "pe"}):
            with mock.patch.dict(os.environ, env, clear=False):
                os.environ.pop("GB_MODE", None)
                os.environ.update(env)
                self.assertEqual(
                    _inmodel_repeats_mode_defaults("gb", 100), (100, 100)
                )
                # lite presets (num_repeat_proposals=2) must stay cheap
                self.assertEqual(
                    _inmodel_repeats_mode_defaults("gb", 2), (2, 2)
                )


# --------------------------------------------------------------------------
# provenance partition
# --------------------------------------------------------------------------
class SurvivorPartitionTest(unittest.TestCase):
    def test_newborn_mask_expression(self):
        # The direct-batch pooling computes the pooled rows' provenance as
        # (~alive_at_pick)[alive_now]: dead at pick + alive now = accepted
        # birth ("newborn"); alive at pick = mature. Dead-now rows never
        # pool.
        alive_at_pick = np.array([True, False, False, True])
        alive_now = np.array([True, True, False, False])
        newborn = (~alive_at_pick)[alive_now]
        np.testing.assert_array_equal(newborn, [False, True])

    def test_split_classes_and_order(self):
        merged = {
            "ids": np.array([10, 11, 12, 13, 14]),
            "specials": np.array([0, 7, 14, 21, 28]),
            "newborn": np.array([False, True, False, True, True]),
        }
        out = _split_by_newborn(merged, np)
        self.assertEqual([name for name, _ in out], ["newborn", "mature"])
        nb, mat = out[0][1], out[1][1]
        # the class key is stripped; row order within a class is preserved
        self.assertNotIn("newborn", nb)
        self.assertNotIn("newborn", mat)
        np.testing.assert_array_equal(nb["ids"], [11, 13, 14])
        np.testing.assert_array_equal(mat["ids"], [10, 12])
        np.testing.assert_array_equal(nb["specials"], [7, 21, 28])
        np.testing.assert_array_equal(mat["specials"], [0, 14])

    def test_split_drops_empty_class(self):
        merged = {
            "ids": np.array([1, 2]),
            "newborn": np.array([False, False]),
        }
        out = _split_by_newborn(merged, np)
        self.assertEqual([name for name, _ in out], ["mature"])
        # all-newborn pool: mature class absent
        merged["newborn"][:] = True
        out = _split_by_newborn(merged, np)
        self.assertEqual([name for name, _ in out], ["newborn"])


class PickedBatchesTest(unittest.TestCase):
    """The staging batch cap (GB_INMODEL_SETUP_BATCH, 2026-08-21): the
    picked pool splits into sequential sub-blocks so the sig-het reference
    stash residency is bounded by the cap instead of the buffer capacity."""

    def _pool(self, n):
        return {
            "ids": np.arange(n) + 100,
            "specials": np.arange(n) * 7,
            "slot_index": np.arange(n)[::-1].copy(),
            "N_vals": np.full(n, 128),
        }

    def test_no_cap_passes_the_pool_through_unsliced(self):
        pool = self._pool(5)
        for cap in (0, 5, 9):
            out = list(_picked_batches(pool, cap))
            self.assertEqual(len(out), 1)
            # identity, not a sliced copy -- the unbatched path is untouched
            self.assertIs(out[0], pool)

    def test_batches_cover_the_pool_in_order(self):
        pool = self._pool(10)
        out = list(_picked_batches(pool, 4))
        self.assertEqual([len(b["ids"]) for b in out], [4, 4, 2])
        for key, full in pool.items():
            np.testing.assert_array_equal(
                np.concatenate([b[key] for b in out]), full
            )
            # every batch carries every key (parallel-array contract)
            for b in out:
                self.assertIn(key, b)


# --------------------------------------------------------------------------
# de-synced accept chain == straight-line reference
# --------------------------------------------------------------------------
_JUMP = np.array([0.6, 0.02, 0.3, 0.3])
_SNR_LIM = 0.9


def _fake_ll(params):
    return -0.5 * ((params - 0.3) ** 2).sum(axis=1) * 10.0


def _fake_hh(params):
    return 4.0 * params[:, 0] ** 2 + 0.2


class _FakePrior:
    def logpdf(self, x):
        ok = (
            (np.abs(x[:, 0]) < 1.0)
            & (x[:, 1] > 2.5)
            & (x[:, 1] < 3.5)
            & (np.abs(x[:, 2]) < 10.0)
            & (np.abs(x[:, 3]) < 10.0)
        )
        return np.where(ok, 0.0, -np.inf)


class _FakePeriodic:
    def wrap(self, d, xp=None):
        return d


class _FakeBuffer:
    def __init__(self, n_slots):
        self.frequency_lims = (
            np.full(n_slots, 2.9e-3),
            np.full(n_slots, 3.1e-3),
        )
        self.opt_snr_rej_samp_limit = _SNR_LIM
        self.snr_rej_detected = False
        self.phase_angle = None
        self.d_h_out = None
        self.h_h_out = None

    def remove_sources_from_band_buffer(self, *a, **k):
        pass

    def add_sources_to_band_buffer(self, *a, **k):
        pass

    def setup_in_model_likelihood(self, *a, **k):
        return False

    def clear_in_model_likelihood(self):
        pass

    def get_add_ll(self, params, slots_in, slots_out, N,
                   phase_maximize=False, leaf_inds=None):
        self.h_h_out = _fake_hh(params)
        self.d_h_out = self.h_h_out * 1.1
        return _fake_ll(params)


class _HarnessMove(GBSpecialStretchMove):
    """In-model repeat harness: real ``_run_in_model_repeats``, fake hooks."""

    def __init__(self):  # bypass the heavy production ctor
        pass

    def _ensure_proposal_tables(self, model, band_sorter):
        pass

    def _debug_seq_select(self, *a, **k):
        return None

    def in_model_proposal(self, coords, chol, band_sorter, source_ids, model):
        self._last_im_kind = "fake"
        return (
            coords + _JUMP[None, :] * np.random.randn(*coords.shape),
            np.zeros(coords.shape[0]),
        )


def _build_problem(n=6, n_src=12, ntemps=2, nwalkers=3, seed=5):
    rng = np.random.RandomState(seed)
    coords = np.zeros((n_src, 4))
    coords[:, 0] = rng.uniform(-0.5, 0.5, n_src)
    coords[:, 1] = rng.uniform(2.95, 3.05, n_src)  # f0 in "mHz" units
    coords[:, 2] = rng.uniform(-1, 1, n_src)
    coords[:, 3] = rng.uniform(-1, 1, n_src)
    ids = np.arange(0, 2 * n, 2)  # subset of the sorter's sources
    picked = {
        "ids": ids,
        "specials": ids * 3,
        "slot_index": np.arange(n, dtype=np.int32),
        "temp_inds": np.arange(n) % ntemps,
        "walker_inds": (np.arange(n) // ntemps) % nwalkers,
        "band_inds": np.arange(n),  # unique band per row -> unique cells
        "N_vals": np.full(n, 64),
    }
    band_sorter = SimpleNamespace(
        inds=np.ones(n_src, dtype=bool),
        coords=coords.copy(),
        leaf_inds=np.arange(n_src),
    )
    band_temps = np.tile(
        np.array([1.0, 0.3])[None, :ntemps], (n, 1)
    )  # (num_bands, ntemps)
    return picked, band_sorter, band_temps, ntemps, nwalkers, n


def _make_move(n_rep):
    m = _HarnessMove()
    m._backend_name = "lisatools_cpu"
    m.use_gpu = False
    m.branch_name = "gb"
    m.name = "fake_inmodel"
    m.num_repeat_proposals = n_rep
    m.use_info_mat_proposal = False
    m.phase_maximize = False
    m.debug = False
    m._f0_col = 1
    m._phi0_col = 3
    m._per_leaf_fill = False
    m.df = 1e-5
    m.sighet_trust_dlna = 0.0
    m.sighet_trust_snr_c = 0.0
    m.sighet_trust_dlna_min = 0.0
    m.sighet_trust_dphase = 0.0
    m.sighet_anchor_check = False
    m.sighet_drift_check = False
    m.sighet_refresh_every = 0
    m.periodic = _FakePeriodic()
    m.gpu_priors = {"gb": _FakePrior()}
    return m


def _reference_run(picked, coords_full, band_temps, n_rep, seed, n_src):
    """The PRE-2026-08-15 straight-line accept chain, verbatim semantics."""
    np.random.seed(seed)
    prior = _FakePrior()
    ids = picked["ids"]
    slots = picked["slot_index"]
    N_vals = picked["N_vals"]
    t_i, w_i, b_i = (
        picked["temp_inds"], picked["walker_inds"], picked["band_inds"]
    )
    n = len(ids)
    beta = band_temps[b_i, t_i]
    curr = coords_full[ids].copy()

    ll_ref = _fake_ll(curr)
    hh0 = _fake_hh(curr)
    curr_prior = prior.logpdf(curr)
    sorter_dh = np.full(n_src, np.nan)
    sorter_hh = np.full(n_src, np.nan)
    sorter_dh[ids] = hh0 * 1.1
    sorter_hh[ids] = hh0

    n4 = (N_vals / 4).astype(int)
    lo_bin = (2.9e-3 / 1e-5) * np.ones(n)
    hi_bin = (3.1e-3 / 1e-5) * np.ones(n)
    lo_bin = lo_bin.astype(int)
    hi_bin = hi_bin.astype(int)

    ntemps, nwalkers = 2, 3
    ll_change = np.zeros((ntemps, nwalkers, n))
    prop = np.zeros((2, ntemps, nwalkers, n), dtype=int)
    acc = np.zeros_like(prop)
    kind = [0, 0, 0, 0]

    for _ in range(n_rep):
        new = curr + _JUMP[None, :] * np.random.randn(*curr.shape)
        new_logp = prior.logpdf(new)
        new_bin = np.abs(new[:, 1] / 1e3 / 1e-5).astype(int)
        new_logp[
            (np.abs(new[:, 1] / 1e3 - curr[:, 1] / 1e3) / 1e-5).astype(int)
            > n4
        ] = -np.inf
        new_logp[new_bin < lo_bin - n4] = -np.inf
        new_logp[new_bin > hi_bin + n4] = -np.inf
        keep = ~np.isinf(new_logp)

        new_ll = np.full(n, -1e300)
        hh_k = dh_k = None
        if keep.any():
            new_ll[keep] = _fake_ll(new[keep])
            hh_k = _fake_hh(new[keep])
            dh_k = hh_k * 1.1
        delta = new_ll - ll_ref
        if keep.any():
            opt = np.sqrt(np.maximum(hh_k, 0.0))
            viol = opt < _SNR_LIM
            if viol.any():
                rows = np.where(keep)[0][viol]
                new_ll[rows] = -1e300
                delta = new_ll - ll_ref

        lnpdiff = beta * delta + (new_logp - curr_prior)
        accept = lnpdiff >= np.log(np.random.rand(n))
        bad = (new_ll <= -1e299) | (new_logp <= -1e229)
        accept[accept & bad] = False

        prop[1][t_i, w_i, b_i] += 1
        kind[0] += n
        kind[2] += int((t_i == 0).sum())
        if accept.any():
            curr[accept] = new[accept]
            ll_ref[accept] = new_ll[accept]
            curr_prior[accept] = new_logp[accept]
            ll_change[t_i[accept], w_i[accept], b_i[accept]] += delta[accept]
            acc[1][t_i[accept], w_i[accept], b_i[accept]] += 1
            kind[1] += int(accept.sum())
            kind[3] += int((t_i[accept] == 0).sum())
            acc_kept = accept[keep]
            sorter_dh[ids[accept]] = dh_k[acc_kept]
            sorter_hh[ids[accept]] = hh_k[acc_kept]

    return {
        "curr": curr, "ll_ref": ll_ref, "ll_change": ll_change,
        "prop": prop, "acc": acc, "kind": kind,
        "sorter_dh": sorter_dh, "sorter_hh": sorter_hh,
    }


class AcceptChainEquivalenceTest(unittest.TestCase):
    def _run_move(self, n_rep, seed, num_repeats=None):
        picked, band_sorter, band_temps, ntemps, nwalkers, n = _build_problem()
        move = _make_move(n_rep)
        buf = _FakeBuffer(n)
        ll_change = np.zeros((ntemps, nwalkers, n))
        prop = np.zeros((2, ntemps, nwalkers, n), dtype=int)
        acc = np.zeros_like(prop)
        np.random.seed(seed)
        move._run_in_model_repeats(
            None, band_sorter, buf, band_temps, picked,
            ll_change, prop, acc, num_repeats=num_repeats,
        )
        return move, band_sorter, picked, band_temps, ll_change, prop, acc

    def test_bit_identical_to_straight_line_reference(self):
        n_rep, seed = 40, 2026
        (move, band_sorter, picked, band_temps,
         ll_change, prop, acc) = self._run_move(n_rep, seed)
        picked_ref, _, band_temps_ref, _, _, n = _build_problem()
        coords0 = _build_problem()[1].coords
        ref = _reference_run(
            picked_ref, coords0, band_temps_ref, n_rep, seed,
            n_src=len(band_sorter.inds),
        )
        # accept DECISIONS + tracked state: exact equality, not allclose
        np.testing.assert_array_equal(
            band_sorter.coords[picked["ids"]], ref["curr"]
        )
        np.testing.assert_array_equal(ll_change, ref["ll_change"])
        np.testing.assert_array_equal(prop, ref["prop"])
        np.testing.assert_array_equal(acc, ref["acc"])
        # per-kind counters aggregate identically (flushed once per block)
        self.assertEqual(move._im_kind_counts["fake"], ref["kind"])
        # sorter d_h/h_h stash follows the accepted rows exactly
        np.testing.assert_array_equal(move._sorter_dh, ref["sorter_dh"])
        np.testing.assert_array_equal(move._sorter_hh, ref["sorter_hh"])
        # sanity: the run actually accepted AND rejected something, and at
        # least one repeat hit the prior/window gate (keep not all-True),
        # otherwise this test exercises nothing.
        n_acc = int(acc[1].sum())
        n_prop = int(prop[1].sum())
        self.assertGreater(n_acc, 0)
        self.assertLess(n_acc, n_prop)

    def test_num_repeats_override_controls_budget(self):
        (move, _, picked, _, _, prop, _) = self._run_move(
            30, 7, num_repeats=5
        )
        # fixed budget: exactly num_repeats proposals per picked source
        self.assertEqual(int(prop[1].sum()), 5 * len(picked["ids"]))

    def test_default_budget_is_num_repeat_proposals(self):
        (move, _, picked, _, _, prop, _) = self._run_move(4, 11)
        self.assertEqual(int(prop[1].sum()), 4 * len(picked["ids"]))


if __name__ == "__main__":
    unittest.main()
