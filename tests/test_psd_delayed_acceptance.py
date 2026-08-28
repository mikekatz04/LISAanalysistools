"""Delayed-acceptance kernel tests for the PSD/GALFOR coarse sidecar (plan-2 T7).

The kernel (`PSDMove._propose_delayed_acceptance`) is exercised on a REAL
PSDMove + eryn stretch machinery, with the fine/coarse likelihood callbacks
monkeypatched to analytic functions so every quantity is checkable:

* Lc == Lf  =>  stage 2 accepts everything and the chain is exactly the
  ordinary fine stretch move with the same RNG stream (bit-equal states);
* the recorded stage-2 ratio equals beta * [(Lf(y)-Lc(y)) - (Lf(x)-Lc(x))]
  recomputed independently from the mocked callbacks;
* stage-2 rejects are reverted bit-exactly; every returned row's log_like is
  the FINE value of its final coordinates (the swap invariant);
* with a deliberately WRONG surrogate, the DA chain still samples the fine
  target (moment + KS comparison against a plain fine chain).
"""

import unittest

import numpy as np
from eryn.model import Model
from eryn.moves.tempering import TemperatureControl
from eryn.state import BranchSupplemental

from lisatools.domains import CoarseWDMSettings, WDMSettings
from lisatools.globalfit.moves.psdmove import PSDMove
from lisatools.globalfit.state import GFState

NW = 6
NDIM = 2
MU = np.array([0.3, -0.2])
SIG = np.array([0.7, 1.3])


def _fine_logl_of(coords_row):
    return float(-0.5 * np.sum(((coords_row - MU) / SIG) ** 2))


def _make_like_fn(bias_fn=None):
    def fn(coords, inds=None, logp=None, supps=None, branch_supps=None):
        c = np.asarray(coords["psd"])[:, :, 0, :]
        out = -0.5 * np.sum(((c - MU) / SIG) ** 2, axis=-1)
        if bias_fn is not None:
            out = out + bias_fn(c)
        if logp is not None:
            out = np.where(np.isinf(np.asarray(logp)), -1e300, out)
        return out, None

    return fn


def _prior_fn(coords, inds=None, supps=None, branch_supps=None):
    c = np.asarray(coords["psd"])[:, :, 0, :]
    inside = np.all(np.abs(c) < 20.0, axis=-1)
    return np.where(inside, 0.0, -np.inf)


class _Base(unittest.TestCase):
    NTEMPS = 2

    def _runtime(self, mode="delayed_acceptance"):
        from lisatools.coarsewdm import CoarseWDMRuntime

        fine = WDMSettings(Nf=8, Nt=10, dt=2.0, force_backend="cpu")
        return CoarseWDMRuntime(
            coarse_settings=CoarseWDMSettings.from_fine(fine, 4),
            use_ws=False,
            mode=mode,
        )

    def _move(self, coarse_fn, fine_fn, seed=7, ntemps=None):
        ntemps = self.NTEMPS if ntemps is None else ntemps
        tc = TemperatureControl(NDIM, NW, ntemps=ntemps, permute=False)
        move = PSDMove(
            None,
            {},
            sampled_branches=["psd"],
            temperature_control=tc,
            coarse_runtime=self._runtime(),
            live_dangerously=True,
            name="da test move",
        )
        move.compute_log_like = fine_fn
        move.compute_coarse_log_like = coarse_fn
        move.compute_log_prior = _prior_fn
        move.periodic = None
        move.accepted = np.zeros((ntemps, NW))
        rng = np.random.RandomState(seed)
        model = Model(None, coarse_fn, _prior_fn, tc, map, rng)
        return move, model, tc

    def _state(self, seed=11, ntemps=None):
        ntemps = self.NTEMPS if ntemps is None else ntemps
        rng = np.random.default_rng(seed)
        coords = {"psd": rng.normal(size=(ntemps, NW, 1, NDIM))}
        supps = BranchSupplemental(
            {"walker_inds": np.tile(np.arange(NW), (ntemps, 1))},
            base_shape=(ntemps, NW),
            copy=True,
        )
        state = GFState(coords, copy=True, supplemental=supps)
        state.log_prior = _prior_fn(coords)
        state.log_like = _make_like_fn()(coords, logp=state.log_prior)[0]
        return state


class DelayedAcceptanceKernelTest(_Base):
    def test_identical_surrogate_reduces_to_fine_move(self):
        """Lc == Lf: DA == the plain fine stretch move, same RNG, bit-equal."""
        fine = _make_like_fn()
        # ntemps=1: the per-repeat identity swap is a no-op, so the extra
        # stage-2 uniforms are the only RNG difference and come AFTER every
        # draw the plain path makes — the single-call states compare exactly.
        move_p, model_p, _ = self._move(fine, fine, seed=99, ntemps=1)
        state_p = self._state(seed=5, ntemps=1)
        # eryn shuffles the red/blue split with GLOBAL np.random
        # (red_blue.py:124) — pin it identically for both paths
        np.random.seed(2024)
        new_p, acc_p = super(PSDMove, move_p).propose(model_p, state_p)

        move_d, model_d, _ = self._move(fine, fine, seed=99, ntemps=1)
        state_d = self._state(seed=5, ntemps=1)
        np.random.seed(2024)
        new_d, keep = move_d._propose_delayed_acceptance(model_d, state_d)

        np.testing.assert_array_equal(np.asarray(acc_p), np.asarray(keep))
        np.testing.assert_array_equal(
            new_p.branches["psd"].coords, new_d.branches["psd"].coords
        )
        # DA's log_like carries the fine values of the final coords
        want = fine({"psd": new_d.branches["psd"].coords})[0]
        np.testing.assert_allclose(new_d.log_like, want, rtol=0, atol=0)

    def test_stage2_ratio_and_revert_bookkeeping(self):
        """log_alpha2 recomputes from the mocks; rejects revert bit-exactly."""
        bias = lambda c: 0.8 * np.sin(3.0 * c.sum(axis=-1))  # noqa: E731
        coarse = _make_like_fn(bias)
        fine = _make_like_fn()
        move, model, tc = self._move(coarse, fine, seed=123)
        state = self._state(seed=21)
        before = np.array(state.branches["psd"].coords, copy=True)
        fine_before = np.array(state.log_like, copy=True)

        new_state, keep = move._propose_delayed_acceptance(model, state)
        dbg = move._da_debug_last
        self.assertIsNotNone(dbg)
        acc = dbg["accepted_stage1"]
        self.assertTrue(acc.any(), "fixture produced no stage-1 acceptances")

        # independent recomputation of the stage-2 ratio on survivor rows,
        # against the PRE-swap final coordinates the stage-2 masks refer to
        final_pre = dbg["coords_final_prewap"]
        betas = np.asarray(tc.betas)[:, None]
        cy = coarse({"psd": final_pre})[0]
        fy = fine({"psd": final_pre})[0]
        # rows that were reverted no longer hold y — recompute on kept rows
        for t, w in zip(*np.where(dbg["keep"])):
            expect = betas[t, 0] * (
                (dbg["fine_y"][t, w] - dbg["coarse_y"][t, w])
                - (dbg["fine_x"][t, w] - dbg["coarse_x"][t, w])
            )
            self.assertAlmostEqual(dbg["log_alpha2"][t, w], expect, places=12)
            self.assertAlmostEqual(fy[t, w], dbg["fine_y"][t, w], places=12)
            self.assertAlmostEqual(cy[t, w], dbg["coarse_y"][t, w], places=12)

        reverted = acc & ~dbg["keep"]
        np.testing.assert_array_equal(final_pre[reverted], before[reverted])
        fine_at_pre = fine({"psd": final_pre})[0]
        np.testing.assert_array_equal(
            fine_at_pre[reverted], fine_before[reverted]
        )
        # fine invariant on EVERY row
        want = fine({"psd": new_state.branches["psd"].coords})[0]
        np.testing.assert_allclose(new_state.log_like, want, rtol=0, atol=0)

    def test_beta_enters_stage_two(self):
        """A cold rung and a beta=0.25 rung scale log_alpha2 by their betas."""
        bias = lambda c: 0.5 * np.tanh(c.sum(axis=-1))  # noqa: E731
        move, model, tc = self._move(_make_like_fn(bias), _make_like_fn(), seed=3)
        state = self._state(seed=8)
        move._propose_delayed_acceptance(model, state)
        dbg = move._da_debug_last
        self.assertIsNotNone(dbg)
        betas = np.asarray(tc.betas)
        self.assertGreater(betas[0], betas[1])
        ratio_core = (dbg["fine_y"] - dbg["coarse_y"]) - (
            dbg["fine_x"] - dbg["coarse_x"]
        )
        np.testing.assert_allclose(
            dbg["log_alpha2"], betas[:, None] * ratio_core, rtol=0, atol=1e-300
        )


class CoarseAuditTest(_Base):
    """The [COARSE_AUDIT] numbers must actually measure surrogate accuracy.

    The stage-2 exponent is beta * [(Lf(y)-Lc(y)) - (Lf(x)-Lc(x))] -- exactly
    the coarse-vs-fine Delta-logL error that scripts/noise/coarse_q_scan.py
    measures offline. So an EXACT surrogate must record |dlogl| == 0 and
    accept every stage-1 survivor at stage 2, and a WRONG surrogate must
    record a non-zero spread. That is what makes the line trustworthy as an
    in-production accuracy readout.
    """

    def _run_one(self, coarse_fn, fine_fn, seed=31):
        move, model, _ = self._move(coarse_fn, fine_fn, seed=seed)
        move._da_audit = []
        state = self._state(seed=seed + 1)
        np.random.seed(4242)
        move._propose_delayed_acceptance(model, state)
        return move._da_audit

    def test_exact_surrogate_reads_zero_error_and_full_acceptance(self):
        fine = _make_like_fn()
        audit = self._run_one(fine, fine)
        self.assertTrue(audit, "no stage-1 survivors: fixture proves nothing")
        s1 = sum(a[0] for a in audit)
        s2 = sum(a[1] for a in audit)
        dd = np.concatenate([a[3] for a in audit])
        self.assertEqual(s2, s1)              # every survivor kept
        self.assertEqual(float(np.max(dd)), 0.0)   # zero measured error

    def test_wrong_surrogate_reads_a_nonzero_error_spread(self):
        bias = lambda c: 0.8 * np.sin(3.0 * c.sum(axis=-1))  # noqa: E731
        audit = self._run_one(_make_like_fn(bias), _make_like_fn())
        self.assertTrue(audit)
        dd = np.concatenate([a[3] for a in audit])
        self.assertGreater(float(np.max(dd)), 1e-3)
        # and the recorded counts stay self-consistent
        for n_s1, n_s2, n_rows, err in audit:
            self.assertLessEqual(n_s2, n_s1)
            self.assertLessEqual(n_s1, n_rows)
            self.assertEqual(err.size, n_s1)


class DelayedAcceptanceTargetsFineTest(_Base):
    NTEMPS = 1

    def _chain(self, da: bool, iters=1500, seed=17):
        bias = lambda c: 0.6 * np.sin(2.0 * c.sum(axis=-1))  # noqa: E731
        coarse = _make_like_fn(bias)
        fine = _make_like_fn()
        move, model, _ = self._move(coarse, fine, seed=seed, ntemps=1)
        state = self._state(seed=seed + 1, ntemps=1)
        samples = []
        np.random.seed(seed + 7)
        for _ in range(iters):
            if da:
                state, _acc = move._propose_delayed_acceptance(model, state)
            else:
                fine_model = Model(None, fine, _prior_fn, model.temperature_control, map, model.random)
                state, _acc = super(PSDMove, move).propose(fine_model, state)
                state.log_like = fine({"psd": state.branches["psd"].coords})[0]
            samples.append(np.array(state.branches["psd"].coords[0, :, 0, :]))
        return np.concatenate(samples[iters // 3 :], axis=0)

    def test_da_samples_the_fine_target_despite_wrong_surrogate(self):
        """Both samplers must satisfy the SAME analytic-target criteria.

        Chain-vs-chain KS on full autocorrelated chains is anti-conservative
        (measured: p ~ 1e-6 between two PLAIN fine chains), so the criteria
        are moments against the analytic target — tolerances calibrated to
        the observed chain-to-chain scatter at this length — plus a KS test
        against the analytic normal on thinned (approximately independent)
        samples. The fine reference chain runs through the identical checks:
        a criterion the exact sampler cannot pass would be a test bug, not a
        kernel bug.
        """
        from scipy.stats import kstest

        for tag, chain in (
            ("da", self._chain(True, iters=3000)),
            ("fine-reference", self._chain(False, iters=3000, seed=1234)),
        ):
            # measured tau ~ 31-35 ITERATIONS for both samplers: thin at the
            # iteration level (the chain is iteration-major over NW walkers)
            thinned = chain.reshape(-1, NW, NDIM)[::60].reshape(-1, NDIM)
            for dim in range(NDIM):
                self.assertLess(
                    abs(np.mean(chain[:, dim]) - MU[dim]),
                    0.10,
                    f"{tag} dim {dim} mean",
                )
                self.assertLess(
                    abs(np.std(chain[:, dim]) / SIG[dim] - 1.0),
                    0.08,
                    f"{tag} dim {dim} std",
                )
                ks = kstest(
                    (thinned[:, dim] - MU[dim]) / SIG[dim], "norm"
                )
                self.assertGreater(
                    ks.pvalue,
                    1e-3,
                    f"{tag} dim {dim}: KS vs analytic target p={ks.pvalue}",
                )


if __name__ == "__main__":
    unittest.main()
