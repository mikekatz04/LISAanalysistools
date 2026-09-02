"""Wiring for the observable-basis in-model proposal.

The MAP is covered by ``test_gb_observable_basis`` and its MEASURE by
``test_gb_observable_basis_invariance`` -- between them the correctness
burden is already discharged. What is left is plumbing, and plumbing is
where this change can still go wrong silently:

* ``factors`` reaching the accept step as a HOST array while everything
  else is CuPy. ``_imk_layout_problem`` checks only dtype and contiguity,
  and ``cupy.float64 is numpy.float64``, so a numpy array passes the gate
  and is then read as a device pointer -- garbage acceptance, no
  exception, plausible-looking chains. The fake-``xp`` test below is the
  only way to catch that without a GPU, and it is the highest-value test
  in this file.
* step scales re-read per repeat instead of snapshotted per block, which
  makes the proposal asymmetric (so ``factors = Jacobian only`` is wrong)
  while leaving the acceptance rate looking perfectly healthy.
* the production path not actually being left alone when the knob is off.

The methods run against a stub ``self``: a real ``GBSpecialStretchMove``
needs data, a backend and GPUs, none of which is under test here. Same
pattern as ``test_gb_inmodel_eigen_axis_wiring``.
"""
import os
import types
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import (
    GBSpecialStretchMove,
    _inmodel_proposal_kind,
)
from lisatools.sampling.gb_observable_basis import fdot_gr

DIST, F0, MC, PHI0, CI, PSI, AL, SD, R = range(9)
NDIM = 9
IN_BASIS = ["dist", "f0", "Mc", "phi0", "cos_iota", "psi", "alpha",
            "sin_delta", "fdot_astro_ratio"]

TOBS = 7.776e6                      # 90 d, the production 3-month run
DF = 1.0 / TOBS

# The real flagship: dist 9.05 kpc, f0 20.3803767 mHz, Mc 0.4658, r ~ 0.
FLAGSHIP = np.array([9.05215813, 20.3803767, 0.465777687, -3.41840873,
                     -0.883190852, 0.38980924, 4.06170662, -0.786384411,
                     0.0])


class _MarkedArray(np.ndarray):
    """Stands in for ``cupy.ndarray``: a type plain numpy never produces."""


class _FakeXP:
    """The numpy namespace, but its outputs are distinguishable.

    Mirrors the one property of the CuPy namespace this file needs: an
    array it produces is a DIFFERENT type from one produced by bare
    ``np``. Anything the proposal builds with bare ``np`` therefore fails
    the type assertion, which on a real GPU is the host-pointer bug.
    """

    def __getattr__(self, k):
        return getattr(np, k)

    def ascontiguousarray(self, a, dtype=None):
        return np.ascontiguousarray(a, dtype=dtype).view(_MarkedArray)

    def asarray(self, a, dtype=None):
        return np.asarray(a, dtype=dtype).view(_MarkedArray)


def _stub(**over):
    """Minimal stand-in carrying only what the methods under test read."""
    s = types.SimpleNamespace(
        xp=np,
        name="rj_fstat_search",
        branch_name="gb",
        _dist_col=DIST, _mc_col=MC, _fdot_astro_col=R, _f0_col=F0,
        _eigen_axis_min_dim=NDIM,
        _eigen_axis_widths_cache=None,
        _observable_map_cache=None,
        _obs_rho=None,
        _last_im_kind=None,
        _last_axis_pick=None,
        _obs_lnfdot=None,
        jump_factor=1.2,
        stretch_probability=0.0,
        time=0,
        df=DF,
        transform_fn=types.SimpleNamespace(input_basis=list(IN_BASIS)),
        _proposal_param_scales=np.ones(NDIM),
        gpu_priors={},
    )
    for k, v in over.items():
        setattr(s, k, v)
    for meth in ("_observable_basis_ready", "_observable_map",
                 "_observable_step_scales", "_observable_proposal",
                 "_eigen_axis_ready", "_eigen_axis_widths"):
        setattr(s, meth, getattr(GBSpecialStretchMove, meth).__get__(s))
    return s


PROPOSE = GBSpecialStretchMove.in_model_proposal
READY = GBSpecialStretchMove._observable_basis_ready
SNAPSHOT = GBSpecialStretchMove._observable_rho_snapshot
SCALES = GBSpecialStretchMove._observable_step_scales


def _coords(n=6, seed=0):
    """``n`` rows scattered around the flagship, all inside the prior box."""
    rng = np.random.default_rng(seed)
    c = np.repeat(FLAGSHIP[None, :], n, axis=0)
    c[:, DIST] *= np.exp(rng.normal(0.0, 0.15, n))
    c[:, F0] += rng.normal(0.0, 1e-4, n)
    c[:, MC] *= np.exp(rng.normal(0.0, 0.05, n))
    c[:, R] += rng.normal(0.0, 0.05, n)
    return c


def _chol(n=6, ndim=NDIM, scale=1e-3):
    """A legacy inverse-information factor: diagonal, distinguishable."""
    c = np.zeros((n, ndim, ndim))
    for i in range(n):
        c[i] = np.diag(scale * (np.arange(ndim) + 1.0))
    return c


# ---------------------------------------------------------------- knobs

class ProposalKindTest(unittest.TestCase):
    """One decision, two spellings; an unknown value must be LOUD."""

    def _kind(self, **env):
        with mock.patch.dict(os.environ, env, clear=False):
            for k in ("GB_INMODEL_PROPOSAL", "GB_INMODEL_OBSERVABLE_BASIS"):
                if k not in env:
                    os.environ.pop(k, None)
            return _inmodel_proposal_kind()

    def test_the_default_is_observable(self):
        """Phase 6: this is now THE in-model proposal for GB.

        Pinned rather than assumed -- an accidental revert of the default
        would silently put every GB run back on the proposal whose f0-fdot
        ridge is 80% too steep, and nothing else would say so.
        """
        self.assertEqual(self._kind(), "observable")

    def test_legacy_remains_reachable_for_the_v7_baseline_comparison(self):
        self.assertEqual(self._kind(GB_INMODEL_PROPOSAL="legacy"), "legacy")

    def test_legacy_and_observable_both_selectable(self):
        self.assertEqual(self._kind(GB_INMODEL_PROPOSAL="legacy"), "legacy")
        self.assertEqual(self._kind(GB_INMODEL_PROPOSAL="observable"),
                         "observable")

    def test_value_is_case_and_whitespace_insensitive(self):
        self.assertEqual(self._kind(GB_INMODEL_PROPOSAL=" Observable "),
                         "observable")

    def test_per_feature_arm_overrides_the_master_switch(self):
        """A runbook that armed the feature explicitly keeps working."""
        self.assertEqual(
            self._kind(GB_INMODEL_PROPOSAL="legacy",
                       GB_INMODEL_OBSERVABLE_BASIS="1"), "observable")
        self.assertEqual(
            self._kind(GB_INMODEL_PROPOSAL="observable",
                       GB_INMODEL_OBSERVABLE_BASIS="0"), "legacy")

    def test_an_unrecognised_value_warns_rather_than_silently_choosing(self):
        """A typo must not quietly select a proposal nobody asked for.

        Env vars are silently ignored when unrecognised (see CLAUDE.md),
        which is how a runbook gets downgraded without anyone noticing.
        """
        with self.assertLogs(
                "lisatools.globalfit.moves.gbspecialstretch", "WARNING"):
            self._kind(GB_INMODEL_PROPOSAL="obserable")


class AmpMaxDecoupledTest(unittest.TestCase):
    """GB_RJ_AMP_MAXIMIZE is its own knob — it must never follow phase max.

    The follow-default is the silent-coupling anti-pattern that bit twice
    in one day: centering-off stripped birth maximization entirely, and
    arming phase max would have silently armed amp max. User ruling
    2026-09-02: separate knobs.
    """

    def _on(self, **env):
        from lisatools.globalfit.moves.gbspecialstretch import (
            _rj_amp_maximize_on)
        with mock.patch.dict(os.environ, env, clear=False):
            if "GB_RJ_AMP_MAXIMIZE" not in env:
                os.environ.pop("GB_RJ_AMP_MAXIMIZE", None)
            return _rj_amp_maximize_on()

    def test_default_off_even_when_phase_max_is_armed(self):
        self.assertFalse(self._on(GB_RJ_PHASE_MAXIMIZE="1"))

    def test_armed_only_by_its_own_env(self):
        self.assertTrue(self._on(GB_RJ_AMP_MAXIMIZE="1"))
        self.assertFalse(self._on(GB_RJ_AMP_MAXIMIZE="0",
                                  GB_RJ_PHASE_MAXIMIZE="1"))


# ----------------------------------------------------------------- gate

class GateTest(unittest.TestCase):

    def test_refuses_a_basis_without_the_observable_columns(self):
        """VGB 5-column and 8-column (A / fdot) bases must stay legacy."""
        with mock.patch.dict(os.environ,
                             {"GB_INMODEL_PROPOSAL": "observable"}):
            for missing in ("_dist_col", "_mc_col", "_fdot_astro_col",
                            "_f0_col"):
                self.assertFalse(READY(_stub(**{missing: None})),
                                 f"{missing}=None must disable the path")
            vgb = _stub(transform_fn=types.SimpleNamespace(
                input_basis=["amp", "f0", "fdot", "phi0", "cos_iota"]))
            self.assertFalse(READY(vgb))

    def test_missing_attribute_entirely_does_not_raise(self):
        s = _stub()
        del s._dist_col
        with mock.patch.dict(os.environ,
                             {"GB_INMODEL_PROPOSAL": "observable"}):
            self.assertFalse(READY(s))

    def test_legacy_disables_it_even_with_a_complete_basis(self):
        with mock.patch.dict(os.environ, {"GB_INMODEL_PROPOSAL": "legacy"}):
            self.assertFalse(READY(_stub()))

    def test_armed_with_a_complete_basis_is_ready(self):
        with mock.patch.dict(os.environ,
                             {"GB_INMODEL_PROPOSAL": "observable"}):
            self.assertTrue(READY(_stub()))


class MapCacheTest(unittest.TestCase):

    def test_tobs_comes_from_df_not_from_basis_settings(self):
        """``_basis_settings.Tobs`` does not exist on ``FDSettings``.

        An unconditional read has already broken every FD-domain GB flow
        once (regression at :11810). Plant a landmine: any attribute read
        on ``_basis_settings`` raises, so a map that survives cannot have
        touched it.
        """
        class Landmine:
            def __getattr__(self, k):
                raise AssertionError(
                    f"_basis_settings.{k} read -- must use 1.0 / self.df")

        s = _stub(_basis_settings=Landmine())
        m = s._observable_map()
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m.Tobs, TOBS, places=3)

    def test_map_is_built_once_and_cached(self):
        s = _stub()
        first = s._observable_map()
        s.transform_fn = None              # would raise if re-resolved
        self.assertIs(s._observable_map(), first)

    def test_an_ineligible_basis_caches_the_refusal_and_warns_once(self):
        s = _stub(transform_fn=types.SimpleNamespace(input_basis=["amp"]))
        with self.assertLogs(
                "lisatools.globalfit.moves.gbspecialstretch", "WARNING"):
            self.assertIsNone(s._observable_map())
        self.assertIsNone(s._observable_map())          # cached, no re-warn


# --------------------------------------------------- rho snapshot / scales

class RhoSnapshotTest(unittest.TestCase):
    """The block-scope snapshot is a CORRECTNESS condition, not a cache."""

    def _snapped(self, n_src=20):
        s = _stub()
        buf = types.SimpleNamespace(
            h_h_out=np.array([100.0, 400.0, 2116.0], dtype=np.complex128))
        SNAPSHOT(s, buf, np.array([3, 7, 11]), n_src)
        return s

    def test_rho_is_sqrt_h_h_scattered_by_source_id(self):
        s = self._snapped()
        np.testing.assert_allclose(
            np.asarray(s._obs_rho)[[3, 7, 11]], [10.0, 20.0, 46.0])

    def test_unvisited_sources_are_not_silently_zero(self):
        """A zero rho would give an INFINITE step, which is not a no-op."""
        s = self._snapped()
        rho = np.asarray(s._obs_rho)
        others = np.delete(rho, [3, 7, 11])
        self.assertTrue(np.all(others > 0.0) or np.all(np.isnan(others)),
                        "unset rows must be nan or positive, never 0")

    def test_step_scales_do_not_move_when_the_live_h_h_changes(self):
        """Re-reading rho per repeat breaks symmetry, invisibly.

        The acceptance rate stays healthy while ``factors = Jacobian
        only`` quietly stops being true, so pin the snapshot directly.
        """
        s = self._snapped()
        ids = np.array([3, 7, 11])
        first = np.asarray(SCALES(s, _chol(3), ids, NDIM))
        s._sorter_hh = np.full(20, 1e9)      # a live update mid-block
        second = np.asarray(SCALES(s, _chol(3), ids, NDIM))
        np.testing.assert_array_equal(first, second)

    def test_scales_never_receive_coords(self):
        """State-dependence belongs in the coordinate change, not here."""
        import inspect
        sig = inspect.signature(GBSpecialStretchMove._observable_step_scales)
        self.assertNotIn("coords", sig.parameters)

    def test_missing_snapshot_falls_back_rather_than_crashing(self):
        s = _stub(_obs_rho=None)
        sc = np.asarray(SCALES(s, _chol(3), np.array([0, 1, 2]), NDIM))
        self.assertEqual(sc.shape, (3, NDIM))
        self.assertTrue(np.all(np.isfinite(sc)) and np.all(sc > 0))

    def test_extrinsic_scales_come_from_the_information_matrix(self):
        """The 5 extrinsic columns are shared between the two bases.

        Their 1-sigma is the legacy marginal ``sqrt(diag(B B^T))`` times
        ``_proposal_param_scales`` -- the same width the production path
        would have used, since those columns pass through unchanged.
        """
        s = self._snapped()
        ch = _chol(3)
        sc = np.asarray(SCALES(s, ch, np.array([3, 7, 11]), NDIM))
        for k, col in enumerate((PHI0, CI, PSI, AL, SD)):
            want = np.sqrt((ch[:, col, :] ** 2).sum(-1))
            np.testing.assert_allclose(sc[:, 3 + k], want, rtol=1e-12)

    def test_chol_none_falls_back_to_the_prior_box(self):
        s = self._snapped()
        sc = np.asarray(SCALES(s, None, np.array([3, 7, 11]), NDIM))
        self.assertTrue(np.all(np.isfinite(sc)) and np.all(sc > 0))


# ------------------------------------------------------------- the draw

class ObservableDrawTest(unittest.TestCase):

    def _run(self, armed=True, n=6, seed=3, **kw):
        coords = _coords(n, seed=seed)
        bs = types.SimpleNamespace(friend_start_inds=None)
        s = _stub(**kw)
        SNAPSHOT(s, types.SimpleNamespace(
            h_h_out=np.full(n, 46.0 ** 2, dtype=np.complex128)),
            np.arange(n), n)
        env = {"GB_INMODEL_PROPOSAL": "observable" if armed else "legacy"}
        with mock.patch.dict(os.environ, env):
            np.random.seed(11)
            new, factors = PROPOSE(s, coords, _chol(n), bs, np.arange(n),
                                   None)
        return s, coords, new, np.asarray(factors)

    def test_kind_is_reported_as_obs_basis(self):
        s, _, _, _ = self._run()
        self.assertEqual(s._last_im_kind, "obs_basis")

    def test_factors_are_finite_and_not_identically_zero(self):
        """A zero factor here means the Jacobian was never applied."""
        _, _, _, f = self._run()
        self.assertTrue(np.all(np.isfinite(f)))
        self.assertGreater(float(np.abs(f).max()), 0.0)

    def test_factors_equal_the_measure_difference(self):
        """``factors`` IS ln|dy/dz|_new - ln|dy/dz|_old, recomputed here."""
        _, c, new, f = self._run()

        def lj(a):
            return (np.log(a[:, DIST])
                    - np.log(fdot_gr(a[:, F0] * 1e-3, a[:, MC])))

        np.testing.assert_allclose(f, lj(new) - lj(c), rtol=1e-12, atol=1e-14)

    def test_factors_route_through_self_xp(self):
        """THE host-pointer test. See the module docstring.

        ``_imk_layout_problem`` cannot tell a host float64 array from a
        device one, so nothing downstream catches this.
        """
        s, _, _, _ = self._run(xp=_FakeXP())
        _, f = PROPOSE, None
        coords = _coords(4)
        bs = types.SimpleNamespace(friend_start_inds=None)
        s2 = _stub(xp=_FakeXP())
        SNAPSHOT(s2, types.SimpleNamespace(
            h_h_out=np.full(4, 2116.0, dtype=np.complex128)), np.arange(4), 4)
        with mock.patch.dict(os.environ,
                             {"GB_INMODEL_PROPOSAL": "observable"}):
            new, fac = PROPOSE(s2, coords, _chol(4), bs, np.arange(4), None)
        self.assertIsInstance(fac, _MarkedArray,
                              "factors must be built with self.xp")

    def test_factors_layout_is_the_accept_kernel_contract(self):
        _, c, _, f = self._run(n=5)
        self.assertEqual(f.dtype, np.float64)
        self.assertEqual(f.ndim, 1)
        self.assertEqual(f.shape[0], c.shape[0])
        self.assertTrue(f.flags.c_contiguous)

    def test_new_coords_are_contiguous_and_writable(self):
        """:12055 and :12242 write into ``new`` in place."""
        _, _, new, _ = self._run()
        self.assertTrue(new.flags.c_contiguous)
        self.assertTrue(new.flags.writeable)

    def test_non_finite_rows_clamp_to_a_finite_sentinel(self):
        """``nan`` comparison need not agree between NumPy and CUDA."""
        s = _stub()
        SNAPSHOT(s, types.SimpleNamespace(
            h_h_out=np.full(2, 1.0, dtype=np.complex128)), np.arange(2), 2)
        c = _coords(2)
        c[0, DIST] = 1e-320          # a step here lands on dist <= 0
        bs = types.SimpleNamespace(friend_start_inds=None)
        with mock.patch.dict(os.environ,
                             {"GB_INMODEL_PROPOSAL": "observable",
                              "GB_INMODEL_OBSERVABLE_JUMP": "50.0"}):
            for _ in range(40):
                _, f = PROPOSE(s, c.copy(), _chol(2), bs, np.arange(2), None)
                self.assertTrue(np.all(np.isfinite(np.asarray(f))))

    def test_the_step_moves_every_sampled_column(self):
        _, c, new, _ = self._run()
        moved = np.abs(new - c) > 0
        for col in range(NDIM):
            if col == MC:
                continue                       # fiber weight defaults to 0
            self.assertTrue(moved[:, col].any(), f"column {col} never moved")

    def test_mc_is_frozen_at_the_default_fiber_weight(self):
        """``dv`` weight 0.0 at first arming: the 8-observable step only."""
        _, c, new, _ = self._run()
        np.testing.assert_array_equal(new[:, MC], c[:, MC])

    def test_fiber_weight_releases_mc(self):
        with mock.patch.dict(
                os.environ, {"GB_INMODEL_OBSERVABLE_FIBER_WEIGHT": "1.0"}):
            _, c, new, _ = self._run()
        self.assertTrue(np.any(new[:, MC] != c[:, MC]))


class RidgeSlopeTest(unittest.TestCase):
    """THE mechanism test: WHICH ridge the proposal walks, not how far.

    The defect is not step size. Measured on the real flagship Fisher
    (``tests/data/flagship_fisher.npz``), the legacy joint draw walks a
    ridge of slope ``d f0 / d fdot = -0.898 T`` where the chirp geometry
    demands ``-T/2``: the mean frequency of a linear chirp is its
    MIDPOINT frequency, so preserving what the data measures requires
    exactly ``df0 = -(T/2) dfdot``. The 0.398 T excess lands as 0.170
    bins of spurious ``f_mid`` motion per fdot step -- about 14 sigma at
    rho = 46, since ``f_mid`` is measured to 0.55/rho = 0.012 bins. That
    penalty, bolted onto every attempt to move ``fdot``, is why ``fdot``
    does not move.

    So a correct-but-useless implementation is one that walks a wrong
    ridge, and slope is what catches it. (An earlier version of this
    class compared ``|d ln fdot|`` magnitudes against a hand-made
    diagonal ``chol``. That comparison was meaningless -- a diagonal
    matrix has no ridge to get wrong.)
    """

    @staticmethod
    def _fdot(a):
        return fdot_gr(a[:, F0] * 1e-3, a[:, MC]) * (1.0 + a[:, R])

    @classmethod
    def _slope(cls, c, new):
        """OLS slope of ``d f0`` [Hz] on ``d fdot`` [Hz/s], in units of T."""
        return float(np.polyfit(cls._fdot(new) - cls._fdot(c),
                                (new[:, F0] - c[:, F0]) * 1e-3, 1)[0]) / TOBS

    def _observable(self, n=20000):
        coords = np.repeat(FLAGSHIP[None, :], n, axis=0)
        bs = types.SimpleNamespace(friend_start_inds=None)
        s = _stub()
        SNAPSHOT(s, types.SimpleNamespace(
            h_h_out=np.full(n, 46.0 ** 2, dtype=np.complex128)),
            np.arange(n), n)
        with mock.patch.dict(os.environ,
                             {"GB_INMODEL_PROPOSAL": "observable"}):
            np.random.seed(5)
            new, _ = PROPOSE(s, coords.copy(), None, bs, np.arange(n), None)
        return coords, np.asarray(new)

    def _legacy(self, n=20000):
        """The PRODUCTION legacy factor: eigh + the relative floor."""
        d = os.path.join(os.path.dirname(__file__), "data",
                         "flagship_fisher.npz")
        if not os.path.exists(d):
            self.skipTest("flagship Fisher fixture not present")
        F = np.load(d)["F"]
        ev, V = np.linalg.eigh(F)
        ev = np.maximum(np.abs(ev), 1e-10 * np.abs(ev).max())
        ch = V / np.sqrt(ev)[None, :]
        rng = np.random.default_rng(1)
        c = np.repeat(FLAGSHIP[None, :], n, axis=0)
        return c, c + (ch @ rng.standard_normal((NDIM, n))).T

    def test_observable_walks_the_analytic_shear_ridge(self):
        """``df0 = -(T/2) dfdot`` by construction, to sampling noise."""
        self.assertAlmostEqual(self._slope(*self._observable()), -0.5,
                               places=2)

    def test_the_legacy_draw_walks_a_measurably_different_ridge(self):
        """Pins the defect. If this ever passes at -0.5, re-derive.

        A legacy slope that matched the geometry would mean the change
        fixes nothing, and this test would be the first to say so.
        """
        self.assertLess(abs(self._slope(*self._legacy()) + 0.5), 1.0)
        self.assertGreater(abs(self._slope(*self._legacy()) + 0.5), 0.2)

    def test_the_observable_ridge_is_closer_to_the_geometry(self):
        obs = abs(self._slope(*self._observable()) + 0.5)
        leg = abs(self._slope(*self._legacy()) + 0.5)
        self.assertLess(obs, 0.1 * leg, f"obs {obs:.4f} vs legacy {leg:.4f}")

    def test_spurious_f_mid_motion_is_bounded_by_its_own_step(self):
        """``f_mid`` must move by its OWN scale, not by the fdot step.

        This is the quantity the likelihood actually pays for: the legacy
        draw spends 0.170 bins of it per fdot step, ~14x the 0.012-bin
        posterior width at rho = 46.
        """
        c, new = self._observable()
        dfd = self._fdot(new) - self._fdot(c)
        df0 = (new[:, F0] - c[:, F0]) * 1e-3
        f_mid_bins = np.std(df0 + 0.5 * TOBS * dfd) * TOBS
        self.assertLess(f_mid_bins, 3.0 * 0.5513 / 46.0)

    def test_ln_fdot_moves_and_cannot_cancel(self):
        """With ``Mc`` frozen, ``r`` carries ``fdot`` alone -- no cancelling.

        The measured legacy failure is that ``Mc`` and ``r`` both drive
        ``fdot`` and move in combinations that leave it put (one
        eigen-axis moves ``r`` by 0.61 and ``ln fdot`` by 0.0062).
        """
        c, new = self._observable()
        d = np.log(self._fdot(new)) - np.log(self._fdot(c))
        # the analytic marginal at rho = 46: sigma = 4.2705 / (rho T^2 fdot)
        want = 4.2705 / (46.0 * TOBS ** 2 * self._fdot(c)[0])
        self.assertAlmostEqual(float(np.std(d)) / want, 1.0, places=1)


class ProductionPathUntouchedTest(unittest.TestCase):
    """Knob off => byte-identical to the pre-change proposal."""

    def _legacy(self, seed=17):
        coords = _coords(6)
        ch = _chol(6)
        bs = types.SimpleNamespace(friend_start_inds=None)
        s = _stub()
        with mock.patch.dict(os.environ, {"GB_INMODEL_PROPOSAL": "legacy"}):
            os.environ.pop("GB_INMODEL_EIGEN_AXIS", None)
            np.random.seed(seed)
            new, f = PROPOSE(s, coords.copy(), ch, bs, np.arange(6), None)
        return s, coords, ch, new, np.asarray(f)

    def test_factors_are_exactly_zero(self):
        _, _, _, _, f = self._legacy()
        np.testing.assert_array_equal(f, np.zeros(6))

    def test_coords_match_the_explicit_legacy_formula(self):
        s, coords, ch, new, _ = self._legacy()
        np.random.seed(17)
        dy = np.einsum("...ij,...j->...i", ch, np.random.randn(*coords.shape))
        want = coords + s.jump_factor * dy * s._proposal_param_scales[None, :]
        np.testing.assert_array_equal(new, want)

    def test_kind_is_still_infomat(self):
        s, _, _, _, _ = self._legacy()
        self.assertEqual(s._last_im_kind, "infomat")

    def test_param_scales_are_not_re_applied_on_the_observable_path(self):
        """The internal basis carries its own scales.

        Multiplying by ``_proposal_param_scales`` again would silently
        shrink ``fdot`` by 1e-16 -- the exact failure mode documented at
        :2631 for ``Mc``.
        """
        coords = _coords(4)
        bs = types.SimpleNamespace(friend_start_inds=None)
        outs = []
        for s_val in (1.0, 1e-16):
            s = _stub(_proposal_param_scales=np.full(NDIM, s_val))
            SNAPSHOT(s, types.SimpleNamespace(
                h_h_out=np.full(4, 2116.0, dtype=np.complex128)),
                np.arange(4), 4)
            with mock.patch.dict(os.environ,
                                 {"GB_INMODEL_PROPOSAL": "observable"}):
                np.random.seed(2)
                new, _ = PROPOSE(s, coords.copy(), None, bs, np.arange(4),
                                 None)
            outs.append(np.asarray(new))
        np.testing.assert_allclose(outs[0], outs[1], rtol=1e-12)


class PeriodicWrapSafetyTest(unittest.TestCase):
    """``periodic.wrap`` runs AFTER ``in_model_proposal`` returns (:12055).

    ``factors`` is therefore evaluated at UNWRAPPED coordinates. That is
    safe only because the measure reads ``dist``, ``f0`` and ``Mc``, none
    of which is periodic -- pin it, because it breaks silently the day
    ``f0`` is added to ``self.periodic``.
    """

    def test_measure_reads_no_periodic_column(self):
        s = _stub()
        m = s._observable_map()
        for name in ("phi0", "psi", "alpha"):
            self.assertNotIn(
                IN_BASIS.index(name),
                {m.dist_index, m.f0_index, m.mc_index},
                f"{name} is periodic and must not enter log_jacobian")

    def test_wrapping_the_angles_does_not_change_factors(self):
        s = _stub()
        m = s._observable_map()
        c = _coords(5)
        new = c.copy()
        new[:, DIST] *= 1.05
        new[:, F0] += 1e-5
        wrapped = new.copy()
        wrapped[:, PHI0] += 2 * np.pi
        wrapped[:, AL] -= 2 * np.pi
        np.testing.assert_allclose(np.asarray(m.factors(c, new)),
                                   np.asarray(m.factors(c, wrapped)),
                                   rtol=1e-14)


class MotionCensusTest(unittest.TestCase):
    """``ln(fdot)`` motion per draw -- the probe gate's headline number.

    A pooled acceptance rate cannot tell "moving well" from "not moving":
    the eigen-axis path reached 67% cold acceptance while its best axis
    moved ``ln(fdot)`` by 0.040 against the 0.35 the flagship needed. So
    the motion is counted separately, proposed and accepted.
    """

    LOGGER = "lisatools.globalfit.moves.gbspecialstretch"

    @staticmethod
    def _pair(n=4):
        c = np.repeat(FLAGSHIP[None, :], n, axis=0)
        new = c.copy()
        new[:, R] += 0.10                       # +10% in the ratio ...
        new[:, F0] += 1e-4                      # ... and f0 moves too
        return c, new

    @classmethod
    def _want_lnfdot(cls):
        """Exact expectation. ``fdot_gr ~ f0^(11/3)``, so moving ``f0``
        contributes on top of the ratio -- an earlier version of this
        test used ``ln(1.10)`` and was short by 11/3 * df0/f0."""
        c, new = cls._pair(1)
        fd = fdot_gr(c[:, F0] * 1e-3, c[:, MC]) * (1 + c[:, R])
        fdn = fdot_gr(new[:, F0] * 1e-3, new[:, MC]) * (1 + new[:, R])
        return abs(float(np.log(fdn[0]) - np.log(fd[0])))

    def _accum(self, n=4, accept=None):
        s = _stub()
        c, new = self._pair(n)
        acc = (np.ones(n, bool) if accept is None else accept)
        GBSpecialStretchMove._obs_motion_accum(s, c, new, acc)
        return s

    def test_accumulates_the_realised_ln_fdot_motion(self):
        s = self._accum()
        n, n_acc, s_fd, s_fd_a, s_fm, s_fm_a = [
            float(v) for v in np.asarray(s._obs_motion)]
        want = self._want_lnfdot()
        self.assertEqual(n, 4.0)
        self.assertEqual(n_acc, 4.0)
        self.assertAlmostEqual(s_fd / n, want, places=12)
        self.assertAlmostEqual(s_fd_a / n_acc, want, places=12)

    def test_rejected_draws_count_as_proposed_but_not_accepted(self):
        s = self._accum(accept=np.array([True, False, False, False]))
        v = [float(x) for x in np.asarray(s._obs_motion)]
        self.assertEqual((v[0], v[1]), (4.0, 1.0))
        self.assertAlmostEqual(v[3] / v[1], self._want_lnfdot(), places=12)

    def test_f_mid_motion_is_reported_in_bins(self):
        """The likelihood pays for ``f_mid``, so that is what to watch."""
        s = self._accum()
        v = np.asarray(s._obs_motion)
        c, new = self._pair(1)
        fd = fdot_gr(c[:, F0] * 1e-3, c[:, MC]) * (1 + c[:, R])
        fdn = fdot_gr(new[:, F0] * 1e-3, new[:, MC]) * (1 + new[:, R])
        want = abs((new[0, F0] - c[0, F0]) * 1e-3
                   + 0.5 * TOBS * (fdn[0] - fd[0])) * TOBS
        self.assertAlmostEqual(float(v[4]) / 4.0, want, places=9)

    def test_report_logs_both_channels_and_resets(self):
        s = self._accum()
        with self.assertLogs(self.LOGGER, "INFO") as cm:
            GBSpecialStretchMove._report_obs_motion(s)
        msg = "\n".join(cm.output)
        self.assertIn("dln_fdot", msg)
        self.assertIn("df_mid", msg)
        self.assertIsNone(s._obs_motion, "must reset after reporting")

    def test_report_is_a_noop_before_any_draw(self):
        s = _stub()
        GBSpecialStretchMove._report_obs_motion(s)      # must not raise

    def test_a_zero_row_batch_does_not_raise(self):
        s = _stub()
        GBSpecialStretchMove._obs_motion_accum(
            s, np.zeros((0, NDIM)), np.zeros((0, NDIM)),
            np.zeros(0, dtype=bool))


class InModelTraceArmTest(unittest.TestCase):
    """A LIVE detailed-balance check on the one term that can be wrong.

    The existing trace prints "(MUST be 0)" only for ``infomat``; without
    an arm here it would print a misleading ``n/a`` for a path whose
    factors are deliberately non-zero.
    """

    LOGGER = "lisatools.globalfit.moves.gbspecialstretch"

    def _trace(self, factors, coords=None, new=None, n=2):
        s = _stub()
        c = _coords(n) if coords is None else coords
        nw = c.copy() if new is None else new
        if new is None:
            nw[:, DIST] *= 1.1
            nw[:, F0] += 2e-5
        z = np.zeros(n)
        with mock.patch.dict(os.environ, {"GB_INMODEL_TRACE": "1"}):
            with self.assertLogs(self.LOGGER, "INFO") as cm:
                GBSpecialStretchMove._inmodel_trace(
                    s, 0, "obs_basis", c, nw, None, factors, np.ones(n),
                    z, z, z, z, z, z, np.zeros(n, dtype=bool),
                    np.zeros(n, int), np.zeros(n, int), np.zeros(n, int),
                    None, None)
        return "\n".join(cm.output)

    def test_a_correct_factor_reports_ok(self):
        c = _coords(2)
        nw = c.copy()
        nw[:, DIST] *= 1.1
        nw[:, F0] += 2e-5
        good = (np.log(nw[:, DIST]) - np.log(fdot_gr(nw[:, F0] * 1e-3,
                                                     nw[:, MC]))
                - np.log(c[:, DIST]) + np.log(fdot_gr(c[:, F0] * 1e-3,
                                                      c[:, MC])))
        msg = self._trace(good, c, nw)
        self.assertIn("obs_basis", msg)
        self.assertNotIn("MISMATCH", msg)

    def test_a_wrong_factor_is_flagged(self):
        """The independent recomputation is the whole value of the arm."""
        msg = self._trace(np.full(2, 0.123456))
        self.assertIn("MISMATCH", msg)

    def test_a_sign_flip_is_flagged(self):
        c = _coords(2)
        nw = c.copy()
        nw[:, DIST] *= 1.1
        nw[:, F0] += 2e-5
        good = (np.log(nw[:, DIST]) - np.log(fdot_gr(nw[:, F0] * 1e-3,
                                                     nw[:, MC]))
                - np.log(c[:, DIST]) + np.log(fdot_gr(c[:, F0] * 1e-3,
                                                      c[:, MC])))
        self.assertIn("MISMATCH", self._trace(-good, c, nw))

    def test_the_trace_never_raises_on_a_surprise(self):
        s = _stub()
        with mock.patch.dict(os.environ, {"GB_INMODEL_TRACE": "1"}):
            GBSpecialStretchMove._inmodel_trace(
                s, 0, "obs_basis", np.zeros((0, NDIM)), None, None, None,
                None, None, None, None, None, None, None, None, None, None,
                None, None, None)


if __name__ == "__main__":
    unittest.main()
