"""Noise-move split (2026-07): PSDMove parameterized over sampled_branches.

Pins the split semantics at the unit level: sampled-vs-fixed coordinate
merging, prior restriction, the sgwb kernel gate, frozen cold rows feeding
the ACA likelihood route, the ladder-mismatch guard in build_noise_moves,
and the full-model cold prior. End-to-end split behavior (independent
ladders on disk, per-branch sub-state write-back) is covered by the
noise_only smoke runs.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    from tests._multishard import FakeMultiShardACA
except ImportError:
    from _multishard import FakeMultiShardACA


def _priors():
    from eryn.prior import ProbDistContainer, uniform_dist

    return {
        "psd": ProbDistContainer(
            {0: uniform_dist(0.0, 1.0), 1: uniform_dist(0.0, 1.0)}
        ),
        "galfor": ProbDistContainer(
            {i: uniform_dist(0.0, 1.0) for i in range(3)}
        ),
        "sgwb": ProbDistContainer(
            {i: uniform_dist(0.0, 1.0) for i in range(2)}
        ),
    }


class _FakeSingleShardACS:
    """Minimal per-walker container ACA for the domain-agnostic route."""

    class _AC:
        def __init__(self):
            self.sens_mat = "original"

    def __init__(self, nwalkers):
        self._acs = [self._AC() for _ in range(nwalkers)]
        self.linear_data_arr = [np.zeros(8, dtype=complex)]
        self.linear_psd_arr = [np.ones(8)]
        self.gpus = None
        self.xp = np
        self.reset_calls = 0

    def __len__(self):
        return len(self._acs)

    def __getitem__(self, i):
        return self._acs[i]

    def reset_linear_psd_arr(self):
        self.reset_calls += 1

    def likelihood(self):
        # distinguishable per-walker values
        return 10.0 * np.arange(len(self._acs), dtype=float)

    def flatten(self):
        return list(self._acs)


class NoiseSplitUnitTest(unittest.TestCase):
    def setUp(self):
        try:
            from lisatools.domains import FDSettings, WDMSettings  # noqa: F401
            from lisatools.globalfit.moves.psdmove import PSDMove
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"psdmove not available: {exc}")
        self.PSDMove = PSDMove
        self.FDSettings = FDSettings
        self.nwalkers = 4
        self.priors = _priors()

    def _move(self, sampled=None, acs=None, basis=None):
        backend = SimpleNamespace(
            use_splines=False,
            basis_settings=basis if basis is not None else self.FDSettings(N=16, df=1e-4),
        )
        return self.PSDMove(
            acs if acs is not None else _FakeSingleShardACS(self.nwalkers),
            self.priors,
            sampled_branches=sampled,
            sensitivity_backend=backend,
            name="noise split test",
        )

    @staticmethod
    def _state(branches, nwalkers, nt=1):
        rng = np.random.default_rng(7)
        coords = {
            b: rng.uniform(0.2, 0.8, size=(nt, nwalkers, 1, nd))
            for b, nd in branches.items()
        }
        return SimpleNamespace(branches_coords=coords)

    # ------------------------------------------------------------------
    # sampled/fixed resolution + merging
    # ------------------------------------------------------------------

    def test_ctor_validates_and_orders(self):
        move = self._move(sampled=["galfor", "psd"])
        self.assertEqual(move.sampled_branches, ["psd", "galfor"])  # canonical
        with self.assertRaises(ValueError):
            self._move(sampled=["psd", "bogus"])

    def test_resolve_sampled_requires_presence(self):
        move = self._move(sampled=["galfor"])
        state = self._state({"psd": 2}, self.nwalkers)  # no galfor branch
        with self.assertRaises(ValueError):
            move._resolve_sampled(state)
        state2 = self._state({"psd": 2, "galfor": 3}, self.nwalkers)
        self.assertEqual(move._resolve_sampled(state2), ["galfor"])
        # default = all present, canonical order
        move_all = self._move(sampled=None)
        self.assertEqual(move_all._resolve_sampled(state2), ["psd", "galfor"])

    def test_merged_rows_sampled_vs_fixed(self):
        move = self._move(sampled=["galfor"])
        nt_mod, nw = 3, self.nwalkers
        cold_psd = np.arange(nw * 2, dtype=float).reshape(nw, 2)
        move._fixed_noise_coords = {"psd": cold_psd}

        rng = np.random.default_rng(3)
        galfor = rng.uniform(size=(nt_mod, nw, 1, 3))
        logp_keep = np.ones((nt_mod, nw), dtype=bool)
        logp_keep[1, 2] = False  # one row dropped by the prior cut
        walker_inds = np.tile(np.arange(nw), (nt_mod, 1))[logp_keep]

        merged = move._merged_noise_rows(
            {"galfor": galfor}, logp_keep, walker_inds
        )
        # sampled branch: proposal rows surviving the cut
        np.testing.assert_array_equal(
            merged["galfor"], galfor[logp_keep][:, 0]
        )
        # fixed branch: the frozen cold row of each row's physical walker
        np.testing.assert_array_equal(merged["psd"], cold_psd[walker_inds])
        self.assertNotIn("sgwb", merged)

    # ------------------------------------------------------------------
    # priors
    # ------------------------------------------------------------------

    def test_log_prior_restricted_to_sampled(self):
        state = self._state({"psd": 2, "galfor": 3}, self.nwalkers, nt=2)
        coords = state.branches_coords

        joint = self._move(sampled=None).compute_log_prior(coords)
        galfor_only = self._move(sampled=["galfor"]).compute_log_prior(coords)
        psd_only = self._move(sampled=["psd"]).compute_log_prior(coords)

        # in-prior uniform draws: each branch contributes 0 here, but the
        # restriction must hold pointwise for the sum decomposition
        np.testing.assert_allclose(joint, galfor_only + psd_only)
        # and the galfor-only move ignores psd coords entirely
        coords_bad_psd = dict(coords)
        coords_bad_psd["psd"] = np.full_like(coords["psd"], 5.0)  # out of prior
        np.testing.assert_allclose(
            self._move(sampled=["galfor"]).compute_log_prior(coords_bad_psd),
            galfor_only,
        )

    def test_cold_noise_log_prior_sums_model(self):
        move = self._move(sampled=["galfor"])
        state = self._state({"psd": 2, "galfor": 3}, self.nwalkers)
        lp = move._cold_noise_log_prior(state)
        self.assertEqual(lp.shape, (self.nwalkers,))
        # all draws in-prior on U(0,1) boxes -> log prior 0 per branch
        np.testing.assert_allclose(lp, 0.0)
        # push one walker's psd out of prior: the FULL-model sum must see it
        state.branches_coords["psd"][0, 1] = 5.0
        lp2 = move._cold_noise_log_prior(state)
        self.assertTrue(np.isneginf(lp2[1]))
        self.assertTrue(np.all(np.isfinite(np.delete(lp2, 1))))

    # ------------------------------------------------------------------
    # kernel gate
    # ------------------------------------------------------------------

    def test_kernel_gate(self):
        from lisatools.domains import WDMSettings  # noqa: F401 (import gate)

        move = self._move(sampled=["psd"])
        self.assertTrue(move._kernel_fast_path_available())
        # sgwb anywhere in the noise MODEL -> kernel path off
        self.assertFalse(move._kernel_fast_path_available(has_sgwb=True))
        # no backend -> off
        move.sensitivity_backend = None
        self.assertFalse(move._kernel_fast_path_available())

    # ------------------------------------------------------------------
    # ACA route consumes frozen cold rows for fixed branches
    # ------------------------------------------------------------------

    def test_aca_route_uses_frozen_cold_rows(self):
        acs = _FakeSingleShardACS(self.nwalkers)
        # WDM-less trick: disable the kernel gate via a non-FD basis
        move = self._move(sampled=["galfor"], acs=acs, basis=object())
        self.assertFalse(move._kernel_fast_path_available())

        cold_psd = 100.0 + np.arange(self.nwalkers * 2, dtype=float).reshape(
            self.nwalkers, 2
        )
        move._fixed_noise_coords = {"psd": cold_psd}

        recorded = []

        def fake_build(w, psd_params, galfor_params, sgwb_params=None):
            recorded.append((int(w), np.array(psd_params), np.array(galfor_params)))
            return f"sens_{w}"

        move._build_sensitivity_for_walker = fake_build

        nt_mod = 2
        rng = np.random.default_rng(11)
        galfor = rng.uniform(0.2, 0.8, size=(nt_mod, self.nwalkers, 1, 3))
        from eryn.state import BranchSupplemental

        supps = BranchSupplemental(
            {"walker_inds": np.tile(np.arange(self.nwalkers), (nt_mod, 1))},
            base_shape=(nt_mod, self.nwalkers),
            copy=True,
        )
        logl, _ = move.compute_log_like({"galfor": galfor}, supps=supps)

        self.assertEqual(logl.shape, (nt_mod, self.nwalkers))
        # every scored row saw ITS walker's frozen psd cold row
        for w, psd_params, galfor_params in recorded:
            np.testing.assert_array_equal(psd_params, cold_psd[w])
        # per-walker likelihoods routed back by walker index
        np.testing.assert_allclose(
            logl, np.tile(10.0 * np.arange(self.nwalkers), (nt_mod, 1))
        )
        # original sens restored + linear buffer reset both times
        self.assertTrue(all(ac.sens_mat == "original" for ac in acs._acs))
        self.assertGreaterEqual(acs.reset_calls, 2)

    def test_coarse_batch_route_uses_exact_frozen_component_cache(self):
        """The full move route caches a frozen branch and preserves lnL."""
        from eryn.state import BranchSupplemental
        from lisatools.coarsewdm import (
            CoarseWDMStatistic,
            coarse_wdm_log_likelihood,
        )
        from lisatools.domains import CoarseWDMSettings, WDMSettings
        from lisatools.sensitivity import CompositeSensitivityBackend

        fine = WDMSettings(
            Nf=128,
            Nt=10,
            dt=5.0,
            min_freq=3e-4,
            max_freq=8e-3,
            force_backend="cpu",
        )
        coarse = CoarseWDMSettings.from_fine(fine, 4)
        shape = coarse.basis_shape_active
        P = np.zeros((3, 3) + shape)
        for channel in range(3):
            P[channel, channel] = 1e-40
        stat = CoarseWDMStatistic(
            P=P,
            Qeff=np.broadcast_to(coarse.cell_sizes, shape).copy(),
            settings=coarse,
        )
        acs = _FakeSingleShardACS(self.nwalkers)
        for ac in acs._acs:
            ac.coarse_stats = stat
        backend = CompositeSensitivityBackend(
            coarse, tdi_generation=2, force_backend="cpu"
        )
        move = self.PSDMove(
            acs,
            self.priors,
            sampled_branches=["galfor"],
            sensitivity_backend=backend,
            name="coarse batch cache test",
        )
        cold_psd = np.tile([15e-12, 3e-15], (self.nwalkers, 1))
        move._fixed_noise_coords = {"psd": cold_psd}
        move._prepare_fixed_component_covariances()
        self.assertEqual(len(move._fixed_component_covariances), self.nwalkers)

        gal0 = np.array(
            [1.17590937048e-44, 2.50409025898e-3, 3.19526170199,
             2.09718967396e-3, 1.10665414857e-3]
        )
        galfor = np.tile(gal0, (2, self.nwalkers, 1, 1))
        galfor[1, :, 0, 0] *= 1.05
        supps = BranchSupplemental(
            {"walker_inds": np.tile(np.arange(self.nwalkers), (2, 1))},
            base_shape=(2, self.nwalkers),
            copy=True,
        )
        selected_widths = []
        original_subband = backend.galfor_coarse_covariance_from_profile

        def record_subband(profile, frequency_indices):
            selected_widths.append(len(frequency_indices))
            return original_subband(profile, frequency_indices)

        with mock.patch.object(
            backend,
            "galfor_coarse_covariance_from_profile",
            side_effect=record_subband,
        ):
            logl, _ = move.compute_log_like(
                {"galfor": galfor},
                logp=np.zeros((2, self.nwalkers)),
                supps=supps,
            )
        expected = np.empty_like(logl)
        for temperature in range(2):
            for walker in range(self.nwalkers):
                matrix = backend(
                    f"fresh_{temperature}_{walker}",
                    cold_psd[walker],
                    galfor_params=galfor[temperature, walker, 0],
                )
                expected[temperature, walker] = coarse_wdm_log_likelihood(
                    stat, matrix
                )
        np.testing.assert_allclose(logl, expected, rtol=2e-15, atol=0.0)
        self.assertTrue(selected_widths)
        self.assertLess(max(selected_widths), fine.Nf_active)
        # The direct batched route never mutates or repacks the ACA.
        self.assertTrue(all(ac.sens_mat == "original" for ac in acs._acs))
        self.assertEqual(acs.reset_calls, 0)

    # ------------------------------------------------------------------
    # threaded per-walker builds (build_threads)
    # ------------------------------------------------------------------

    def _aca_route_run(self, build_threads):
        """Score one ACA-route proposal block; return (logl, per-walker builds)."""
        import threading

        acs = _FakeSingleShardACS(self.nwalkers)
        move = self.PSDMove(
            acs,
            self.priors,
            sampled_branches=["galfor"],
            sensitivity_backend=SimpleNamespace(
                use_splines=False, basis_settings=object()  # non-FD -> ACA route
            ),
            build_threads=build_threads,
            name="threaded build test",
        )
        move._fixed_noise_coords = {
            "psd": 100.0
            + np.arange(self.nwalkers * 2, dtype=float).reshape(self.nwalkers, 2)
        }

        recorded = []
        lock = threading.Lock()

        def fake_build(w, psd_params, galfor_params, sgwb_params=None):
            with lock:
                recorded.append((int(w), np.array(galfor_params)))
            return f"sens_{w}"

        move._build_sensitivity_for_walker = fake_build

        nt_mod = 3
        rng = np.random.default_rng(11)
        galfor = rng.uniform(0.2, 0.8, size=(nt_mod, self.nwalkers, 1, 3))
        from eryn.state import BranchSupplemental

        supps = BranchSupplemental(
            {"walker_inds": np.tile(np.arange(self.nwalkers), (nt_mod, 1))},
            base_shape=(nt_mod, self.nwalkers),
            copy=True,
        )
        logl, _ = move.compute_log_like({"galfor": galfor}, supps=supps)
        return move, acs, logl, sorted(recorded, key=lambda r: (r[0], r[1].tobytes()))

    def test_threaded_builds_match_serial(self):
        """build_threads>1 must not change what is built or scored."""
        _, acs1, logl1, rec1 = self._aca_route_run(1)
        _, acs4, logl4, rec4 = self._aca_route_run(4)

        np.testing.assert_array_equal(logl1, logl4)
        self.assertEqual(len(rec1), len(rec4))
        for (w1, g1), (w4, g4) in zip(rec1, rec4):
            self.assertEqual(w1, w4)
            np.testing.assert_array_equal(g1, g4)
        # bookkeeping is unaffected: originals restored, buffers reset
        for acs in (acs1, acs4):
            self.assertTrue(all(ac.sens_mat == "original" for ac in acs._acs))

    def test_build_threads_default_is_serial(self):
        move = self._move(sampled=["galfor"])
        self.assertEqual(move.build_threads, 1)
        self.assertIsNone(move._build_pool)

    def test_first_build_is_serial_then_threaded(self):
        """The cold backend cache is filled exactly once, unraced."""
        move, _, _, _ = self._aca_route_run(4)
        self.assertTrue(move._build_warmed)

    def test_thread_pool_never_enters_a_copy(self):
        """Sprint deepcopy/pickle rule: the pool must not ride along.

        Asserted on ``__getstate__`` directly rather than on a full
        ``deepcopy(move)`` -- the move also holds the ACA, which is
        unpicklable by design (nanobind wraps; ``_FakeSingleShardACS`` stands
        in for that here by holding a module). ``__getstate__`` is what both
        ``copy.deepcopy`` and ``pickle`` route through, so this pins the
        contract at the point the rule cares about.
        """
        move, _, _, _ = self._aca_route_run(4)
        self.assertIsNotNone(move._build_pool)  # the run created one

        state = move.__getstate__()
        self.assertIsNone(state["_build_pool"])
        self.assertEqual(state["_build_threads"], 4)  # knob survives
        self.assertIsNotNone(move._build_pool)  # live move keeps its pool

        # a restored copy rebuilds lazily
        restored = object.__new__(type(move))
        restored.__dict__.update(state)
        self.assertIsNone(restored._build_pool)
        self.assertIsNotNone(restored.build_pool)
        restored.build_pool.shutdown(wait=False)

    # ------------------------------------------------------------------
    # linear_psd_arr repack skip (CPU, no C++ kernel consumer)
    # ------------------------------------------------------------------

    def _move_with_backend(self, backend_name, dcga=None):
        backend = SimpleNamespace(
            use_splines=False,
            basis_settings=object(),  # non-FD -> ACA route
            backend=SimpleNamespace(name=backend_name),
        )
        return self.PSDMove(
            _FakeSingleShardACS(self.nwalkers),
            self.priors,
            sampled_branches=["galfor"],
            sensitivity_backend=backend,
            dcga=dcga,
            name="repack gate test",
        )

    def test_repack_skipped_on_cpu_only(self):
        """The skip is CPU-only and never applies when a DCGA reads the buffer."""
        self.assertTrue(self._move_with_backend("lisatools_cpu")._skip_linear_psd_repack)
        for gpu_name in ("lisatools_cuda12x", "lisatools_cuda11x", "lisatools_cuda13x"):
            self.assertFalse(
                self._move_with_backend(gpu_name)._skip_linear_psd_repack,
                f"{gpu_name} must keep repacking",
            )
        # a DCGA consumer wins over the backend name
        self.assertFalse(
            self._move_with_backend(
                "lisatools_cpu", dcga=SimpleNamespace()
            )._skip_linear_psd_repack
        )
        # unknown/missing backend => conservative (repack)
        move = self._move_with_backend("lisatools_cpu")
        move.sensitivity_backend = SimpleNamespace(use_splines=False, basis_settings=object())
        self.assertFalse(move._skip_linear_psd_repack)

    def test_skipping_the_repack_does_not_change_scores(self):
        """WARNING guard: the skip is only valid while nothing reads the buffer.

        The ACA route reads ``ac.sens_mat.invC`` per container, so skipping the
        repack must be score-neutral. If a WDM C++ likelihood kernel is added
        that reads ``linear_psd_arr``, this stops being true -- see the warning
        on ``PSDMove._skip_linear_psd_repack``.
        """
        results = {}
        for backend_name in ("lisatools_cpu", "lisatools_cuda12x"):
            acs = _FakeSingleShardACS(self.nwalkers)
            move = self._move_with_backend(backend_name)
            move.acs = acs
            move._fixed_noise_coords = {
                "psd": 100.0
                + np.arange(self.nwalkers * 2, dtype=float).reshape(self.nwalkers, 2)
            }
            move._build_sensitivity_for_walker = (
                lambda w, p, g, s=None: f"sens_{w}"
            )
            nt_mod = 2
            rng = np.random.default_rng(5)
            galfor = rng.uniform(0.2, 0.8, size=(nt_mod, self.nwalkers, 1, 3))
            from eryn.state import BranchSupplemental

            supps = BranchSupplemental(
                {"walker_inds": np.tile(np.arange(self.nwalkers), (nt_mod, 1))},
                base_shape=(nt_mod, self.nwalkers),
                copy=True,
            )
            logl, _ = move.compute_log_like({"galfor": galfor}, supps=supps)
            results[backend_name] = (logl, acs.reset_calls)

        np.testing.assert_array_equal(
            results["lisatools_cpu"][0], results["lisatools_cuda12x"][0]
        )
        # and the CPU path really did skip the repacks the GPU path performed
        self.assertEqual(results["lisatools_cpu"][1], 0)
        self.assertGreater(results["lisatools_cuda12x"][1], 0)

    # ------------------------------------------------------------------
    # builder-level ladder-mismatch guard
    # ------------------------------------------------------------------

    def test_build_noise_moves_ladder_mismatch_guard(self):
        from lisatools.globalfit.recipe import build_noise_moves

        general_info = SimpleNamespace(nwalkers=4, ntemps=1, gpus=None)
        curr = SimpleNamespace(
            general_info=general_info,
            source_info={
                "psd": SimpleNamespace(
                    betas=np.ones(8), ntemps=8, transform=None,
                    num_prop_repeats=5,
                ),
                "galfor": SimpleNamespace(
                    betas=np.ones(4), ntemps=4, transform=None,
                    num_prop_repeats=5,
                ),
            },
        )
        engine_info = SimpleNamespace(ndims={"psd": 2, "galfor": 5})
        acs = _FakeSingleShardACS(4)

        with self.assertRaises(ValueError) as ctx:
            build_noise_moves(
                engine_info, curr, acs, self.priors,
                sampled_branches=["psd", "galfor"],
            )
        msg = str(ctx.exception)
        self.assertIn("GALFOR_NTEMPS", msg)
        self.assertIn("ntemps=8", msg)
        self.assertIn("ntemps=4", msg)

        with self.assertRaises(ValueError):
            build_noise_moves(
                engine_info, curr, acs, self.priors,
                sampled_branches=["sgwb"],  # not in the run
            )


if __name__ == "__main__":
    unittest.main()
